import sys
from time import time
import random
import numpy as np
import numba as nb
import numba.types as nt
from collections import defaultdict, namedtuple
from worms.util import contig_idx_breaks, jit, InProcessExecutor, NonFuture
import concurrent.futures as cf
from tqdm import tqdm
from worms.database import SpliceDB

try:
    # this is such bullshit...
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
    HAVE_PYROSETTA = True
except ImportError:
    HAVE_PYROSETTA = False


def Edge(u, ublks, v, vblks, verbosity=0, **kw):

    splices, nout, nent = get_allowed_splices(
        u, ublks, v, vblks, verbosity=verbosity, **kw
    )
    maxentries = max(len(_) for _ in splices)
    splice_ary = np.zeros((len(splices), maxentries + 1), dtype=np.int32) - 1
    for i, a in enumerate(splice_ary):
        a[0] = len(splices[i]) + 1
        a[1:a[0]] = sorted(splices[i])

    assert np.max(splice_ary[:, 1:]) < len(v.inbreaks), \
            'egde.py bad splice_ary'
    assert len(splice_ary) == 1 + np.max(u.inout[:, 1]), \
            'edge.py, bad splice_ary'

    return _Edge(splice_ary, nout, nent)


def splice_metrics_pair(
        blk0,
        blk1,
        splice_max_rms,
        splice_clash_d2,
        splice_contact_d2,
        splice_rms_range,
        splice_clash_contact_range,
        skip_on_fail,
):
    return _jit_splice_metrics(
        blk0.chains, blk1.chains, blk0.ncac, blk1.ncac, blk0.stubs, blk1.stubs,
        blk0.connections, blk1.connections, splice_clash_d2, splice_contact_d2,
        splice_rms_range, splice_clash_contact_range, splice_max_rms,
        skip_on_fail
    )


def get_allowed_splices(
        u,
        ublks,
        v,
        vblks,
        splicedb=None,
        splice_max_rms=0.7,
        splice_ncontact_cut=30,
        splice_clash_d2=4.0**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_rms_range=6,
        splice_clash_contact_range=60,
        skip_on_fail=True,
        parallel=False,
        verbosity=1,
        cache_sync=0.001,
        precache_splices=False,
        **kw
):
    assert (u.dirn[1] + v.dirn[0]) == 1, 'get_allowed_splices dirn mismatch'

    # note: this is duplicated in edge_batch.py
    params = (
        splice_max_rms, splice_ncontact_cut, splice_clash_d2,
        splice_contact_d2, splice_rms_range, splice_clash_contact_range
    )

    outidx = _get_outidx(u.inout[:, 1])
    outblk = u.ibblock[outidx]
    outres = u.ires[outidx, 1]

    inblk = v.ibblock[v.inbreaks[:-1]]
    inres = v.ires[v.inbreaks[:-1], 0]
    inblk_breaks = contig_idx_breaks(inblk)

    outblk_res = defaultdict(list)
    for iblk, ires in zip(outblk, outres):
        outblk_res[iblk].append(ires)
    for iblk in outblk_res.keys():
        outblk_res[iblk] = np.array(outblk_res[iblk], 'i4')

    inblk_res = defaultdict(list)
    for iblk, ires in zip(inblk, inres):
        inblk_res[iblk].append(ires)
    for iblk in inblk_res.keys():
        inblk_res[iblk] = np.array(inblk_res[iblk], 'i4')
        assert np.all(sorted(inblk_res[iblk]) == inblk_res[iblk])

    nout = sum(len(a) for a in outblk_res.values())
    nent = sum(len(a) for a in inblk_res.values())
    valid_splices = [list() for i in range(nout)]

    swapped = False
    if u.dirn[1] == 0:  # swap so N-to-C!
        swapped = True
        u, ublks, v, vblks = v, vblks, u, ublks
        outblk_res, inblk_res = inblk_res, outblk_res
        outblk, inblk = inblk, outblk

    pairs_with_no_valid_splices = 0
    tcache = 0

    exe = InProcessExecutor()
    if parallel:
        exe = cf.ProcessPoolExecutor(max_workers=parallel)
    # exe = cf.ThreadPoolExecutor(max_workers=parallel) if parallel else InProcessExecutor()
    with exe as pool:
        futures = list()
        ofst0 = 0
        for iblk0, ires0 in outblk_res.items():
            blk0 = ublks[iblk0]
            key0 = blk0.filehash
            t = time()
            cache = splicedb.partial(params, key0) if splicedb else None
            tcache += time() - t
            ofst1 = 0
            for iblk1, ires1 in inblk_res.items():
                blk1 = vblks[iblk1]
                key1 = blk1.filehash
                if cache and key1 in cache and cache[key1]:
                    splices = cache[key1]
                    future = NonFuture(splices, dummy=True)
                else:
                    future = pool.submit(
                        _jit_splice_metrics, blk0.chains, blk1.chains,
                        blk0.ncac, blk1.ncac, blk0.stubs, blk1.stubs,
                        blk0.connections, blk1.connections, splice_clash_d2,
                        splice_contact_d2, splice_rms_range,
                        splice_clash_contact_range, splice_max_rms,
                        skip_on_fail
                    )
                fs = (iblk0, iblk1, ofst0, ofst1, ires0, ires1)
                future.stash = fs
                futures.append(future)
                ofst1 += len(ires1)
            ofst0 += len(ires0)

        if verbosity > 0 and tcache > 1.0:
            print('get_allowed_splices read caches time:', tcache)

        future_iter = cf.as_completed(futures)
        if verbosity > 1 and not precache_splices:
            future_iter = tqdm(
                cf.as_completed(futures),
                'checking splices',
                total=len(futures)
            )
        for future in future_iter:
            iblk0, iblk1, ofst0, ofst1, ires0, ires1 = future.stash
            result = future.result()
            if len(result) is 3 and isinstance(result[0], np.ndarray):
                rms, nclash, ncontact = result
                ok = ((nclash == 0) * (rms <= splice_max_rms) *
                      (ncontact >= splice_ncontact_cut))
                result = _splice_respairs(ok, ublks[iblk0], vblks[iblk1])
                if np.sum(ok) == 0:
                    # print(nclash)
                    print('N no clash', np.sum(nclash == 0))
                    # print(rms)
                    print('N rms', np.sum(rms <= splice_max_rms))
                    # print(ncontact)
                    print('N contact', np.sum(ncontact >= splice_ncontact_cut))

                if splicedb:
                    key0 = ublks[iblk0].filehash  # C-term side
                    key1 = vblks[iblk1].filehash  # N-term side
                    splicedb.add(params, key0, key1, result)
                    if np.random.random() < cache_sync:
                        print('sync_to_disk splices data')
                        splicedb.sync_to_disk()

            if swapped:
                result = result[1], result[0]
                ires0, ires1 = ires1, ires0
                ofst0, ofst1 = ofst1, ofst0

            if len(result[0]) == 0:
                pairs_with_no_valid_splices += 1
                continue
            index_of_ires0 = _index_of_map(ires0, np.max(result[0]))
            index_of_ires1 = _index_of_map(ires1, np.max(result[1]))
            irs = index_of_ires0[result[0]] + ofst0
            jrs = index_of_ires1[result[1]] + ofst1
            for ir, jr in zip(irs, jrs):
                valid_splices[ir].append(jr)

    if cache_sync > 0 and splicedb:
        splicedb.sync_to_disk()

    if pairs_with_no_valid_splices:
        print(
            'pairs with no valid splices: ', pairs_with_no_valid_splices, 'of',
            len(outblk_res) * len(inblk_res)
        )

    return valid_splices, nout, nent




@nb.jitclass((
    ('splices', nt.int32[:, :]),
    ('nout'   , nt.int32),
    ('nent'   , nt.int32),
))  # yapf: disable
class _Edge:
    """contains junction scores
    """

    def __init__(self, splices, nout, nent):
        self.splices = splices
        self.nout = nout
        self.nent = nent

    @property
    def len(self):
        return len(self.splices)

    def allowed_entries(self, i):
        assert i >= 0, 'edge.py allowed_entries bad i'
        assert self.splices.shape[0] > i, 'edge.py allowed_entries bad i'
        assert self.splices.shape[1] >= self.splices[i, 0], \
            'edge.py allowed_entries bad i'

        return self.splices[i, 1:self.splices[i, 0]]

    def total_allowed_splices(self):
        return np.sum(self.splices[:, 0]) - len(self.splices)

    @property
    def _state(self):
        return (self.splices, self.nout, self.nent)


@jit
def _chainbounds_of_ires(chains, ires):
    for c in range(len(chains)):
        if chains[c, 0] <= ires < chains[c, 1]:
            return chains[c, 0], chains[c, 1]
    return (-1, -1)


@jit
def _ires_from_conn(conn, dirn):
    n = 0
    for i in range(len(conn)):
        if conn[i, 0] == dirn:
            n += conn[i, 1] - 2
    ires = np.empty(n, dtype=np.int32)
    pos = 0
    for i in range(len(conn)):
        if conn[i, 0] == dirn:
            ires[pos:pos + conn[i, 1] - 2] = conn[i, 2:conn[i, 1]]
            pos += conn[i, 1] - 2
    assert pos == n
    return ires


@jit
def _index_of_map(ary, mx):
    map = -np.ones(mx + 1, dtype=np.int32)
    for i, v in enumerate(ary):
        if v <= mx:
            map[v] = i
    return map


@jit
def _jit_splice_metrics(chains0, chains1,
                        ncac0_3d, ncac1_3d,
                        stubs0, stubs1,
                        conn0, conn1,
                        splice_clash_d2,
                        splice_contact_d2,
                        splice_rms_range,
                        splice_clash_contact_range,
                        splice_max_rms,
                        skip_on_fail=True):  # yapf: disable

    aln0s = _ires_from_conn(conn0, 1)
    aln1s = _ires_from_conn(conn1, 0)

    out_rms = np.zeros((len(aln0s), len(aln1s)), dtype=np.float32)
    out_nclash = -np.ones((len(aln0s), len(aln1s)), dtype=np.float32)
    out_ncontact = -np.ones((len(aln0s), len(aln1s)), dtype=np.float32)

    ncac0 = ncac0_3d.reshape(-1, 4)
    ncac1 = ncac1_3d.reshape(-1, 4)

    for ialn1, aln1 in enumerate(aln1s):
        chainb10, chainb11 = _chainbounds_of_ires(chains1, aln1)
        if np.abs(chainb10 - aln1) < splice_rms_range: continue
        if np.abs(chainb11 - aln1) <= splice_rms_range: continue
        stub1_inv = np.linalg.inv(stubs1[aln1])

        for ialn0, aln0 in enumerate(aln0s):
            chainb00, chainb01 = _chainbounds_of_ires(chains0, aln0)
            if np.abs(chainb00 - aln0) < splice_rms_range: continue
            if np.abs(chainb01 - aln0) <= splice_rms_range: continue
            xaln = stubs0[aln0] @ stub1_inv

            sum_d2, n1b = 0.0, 0
            for i in range(-3 * splice_rms_range, 3 * splice_rms_range + 3):
                a = ncac0[3 * aln0 + i]
                b = xaln @ ncac1[3 * aln1 + i]
                sum_d2 += np.sum((a - b)**2)
            rms = np.sqrt(sum_d2 / (splice_rms_range * 6 + 3))
            assert 0 <= rms < 9e9, 'bad rms'
            out_rms[ialn0, ialn1] = rms

            if skip_on_fail and rms > splice_max_rms:
                continue

            nclash, ncontact = 0, 0
            for j in range(4, 3 * splice_clash_contact_range + 3, 3):
                if 3 * aln1 + j >= len(ncac1): continue
                b = xaln @ ncac1[3 * aln1 + j]
                for i in range(-2, -3 * splice_clash_contact_range - 1, -3):
                    if 3 * aln0 + i < 0: continue
                    a = ncac0[3 * aln0 + i]
                    d2 = np.sum((a - b)**2)
                    if d2 < splice_clash_d2:
                        nclash += 1
                    elif i % 3 == 1 and j % 3 == 1 and d2 < splice_contact_d2:
                        ncontact += 1
            assert 0 <= np.isnan(nclash) < 99999, 'bad nclash'
            assert 0 <= np.isnan(ncontact) < 99999, 'bad ncontact'
            out_nclash[ialn0, ialn1] = nclash
            out_ncontact[ialn0, ialn1] = ncontact

    return out_rms, out_nclash, out_ncontact


@jit
def _check_inorder(ires):
    for i in range(len(ires) - 1):
        if ires[i] > ires[i + 1]:
            return False
    return True


@jit
def _get_outidx(iout):
    outidx = np.empty(np.max(iout) + 1, dtype=np.int32)
    for i, o in enumerate(iout):
        outidx[o] = i
    return outidx


@jit
def _splice_respairs(edgemat, bbc, bbn):
    n = np.sum(edgemat)
    out0 = np.empty(n, dtype=np.int32)
    out1 = np.empty(n, dtype=np.int32)
    res0 = _ires_from_conn(bbc.connections, 1)
    res1 = _ires_from_conn(bbn.connections, 0)
    assert len(res0) == edgemat.shape[0]
    assert len(res1) == edgemat.shape[1]
    count = 0
    for i in range(edgemat.shape[0]):
        for j in range(edgemat.shape[1]):
            if edgemat[i, j]:
                out0[count] = res0[i]
                out1[count] = res1[j]
                count += 1
    assert count == n
    return out0, out1
