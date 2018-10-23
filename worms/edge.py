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

    if splice_ary.shape[1] > 1:
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
        splice_clash_contact_by_helix,
        skip_on_fail,
):
    return _jit_splice_metrics(
        blk0.chains, blk1.chains, blk0.ncac, blk1.ncac, blk0.stubs, blk1.stubs,
        blk0.connections, blk1.connections, blk0.ss, blk1.ss, blk0.cb, blk1.cb,
        splice_clash_d2, splice_contact_d2, splice_rms_range,
        splice_clash_contact_range, splice_clash_contact_by_helix,
        splice_max_rms, skip_on_fail
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
        splice_clash_contact_by_helix=True,
        splice_ncontact_no_helix_cut=0,
        splice_nhelix_contacted_cut=0,
        skip_on_fail=True,
        parallel=False,
        verbosity=1,
        cache_sync=0.001,
        precache_splices=False,
        pbar=False,
        pbar_interval=10.0,
        **kw
):
    assert (u.dirn[1] + v.dirn[0]) == 1, 'get_allowed_splices dirn mismatch'

    # note: this is duplicated in edge_batch.py and they need to be the same
    params = (
        splice_max_rms, splice_ncontact_cut, splice_clash_d2,
        splice_contact_d2, splice_rms_range, splice_clash_contact_range,
        splice_clash_contact_by_helix, splice_ncontact_no_helix_cut,
        splice_nhelix_contacted_cut
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
                        blk0.connections, blk1.connections, blk0.ss, blk1.ss,
                        blk0.cb, blk1.cb, splice_clash_d2, splice_contact_d2,
                        splice_rms_range, splice_clash_contact_range,
                        splice_clash_contact_by_helix, splice_max_rms,
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
        if pbar and not precache_splices:
            future_iter = tqdm(
                cf.as_completed(futures),
                'checking splices',
                mininterval=pbar_interval,
                total=len(futures)
            )
        for future in future_iter:
            iblk0, iblk1, ofst0, ofst1, ires0, ires1 = future.stash
            result = future.result()
            if len(result) is 5 and isinstance(result[0], np.ndarray):
                rms, nclash, ncontact, ncnh, nhc = result
                ok = ((nclash == 0) * (rms <= splice_max_rms) *
                      (ncontact >= splice_ncontact_cut) *
                      (ncnh >= splice_ncontact_no_helix_cut) *
                      (nhc >= splice_nhelix_contacted_cut))
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

    @property
    def memuse(self):
        return self.splices.size * self.splices.itemsize


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
def _helix_range(ss):
    helixof = np.zeros_like(ss, dtype=np.int32) - 1
    nhelix = 0
    prevh = 0
    # 72 is 'H'
    for i in range(len(ss)):
        if ss[i] == 72 and prevh == 0:
            nhelix += 1
        prevh = ss[i] == 72
    hrange = np.zeros((nhelix, 2), dtype=np.int32)
    nhelix = 0
    for i in range(len(ss)):
        if ss[i] == 72 and prevh == 0:  # starth
            hrange[nhelix, 0] = i
        elif ss[i] != 72 and prevh == 1:  # endh
            hrange[nhelix, 1] = i
            nhelix += 1
        if ss[i] == 72:
            helixof[i] = nhelix
        prevh = ss[i] == 72
    return hrange, helixof


@jit
def _jit_splice_metrics(chains0, chains1,
                        ncac0_3d, ncac1_3d,
                        stubs0, stubs1,
                        conn0, conn1,
                        ss0, ss1,
                        cb0, cb1,
                        splice_clash_d2,
                        splice_contact_d2,
                        splice_rms_range,
                        splice_clash_contact_range,
                        splice_clash_contact_by_helix,
                        splice_max_rms,
                        skip_on_fail=True):  # yapf: disable


    aln0s = _ires_from_conn(conn0, 1)
    aln1s = _ires_from_conn(conn1, 0)

    outshape = len(aln0s), len(aln1s)
    out_rms = np.zeros(outshape, dtype=np.float32)
    out_nclash = -np.ones(outshape, dtype=np.float32)
    out_ncontact = -np.ones(outshape, dtype=np.int32)
    out_ncnh = -np.ones(outshape, dtype=np.int32)
    out_nhc = -np.ones(outshape, dtype=np.int32)

    ncac0 = ncac0_3d.reshape(-1, 4)
    ncac1 = ncac1_3d.reshape(-1, 4)

    hrange0, helixof0 = _helix_range(ss0)
    hrange1, helixof1 = _helix_range(ss1)

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

            if splice_clash_contact_by_helix:
                nclash, ncontact, ncnh, nhc = _clash_contact_by_helix(
                    aln0, aln1, xaln, cb0, cb1, hrange0, hrange1, helixof0,
                    helixof1, splice_clash_d2, splice_contact_d2
                )
            else:
                nclash, ncontact, ncnh, nhc = _clash_contact_simple(
                    aln0, aln1, xaln, ncac0, ncac1, splice_clash_contact_range,
                    splice_clash_d2, splice_contact_d2
                )
            out_nclash[ialn0, ialn1] = nclash
            out_ncontact[ialn0, ialn1] = ncontact
            out_ncnh[ialn0, ialn1] = ncnh
            out_nhc[ialn0, ialn1] = nhc

    return out_rms, out_nclash, out_ncontact, out_ncnh, out_nhc


@jit
def _mark_cont_aln(
        ct0, ct1, bnd0, bnd1, cb0, cb1, xaln, mark0, mark1, cld2, ctd2
):
    clash, contact = False, False
    for jr in range(*bnd1):
        cbjr = xaln @ cb1[jr]
        for ir in range(*bnd0):
            cbir = cb0[ir]
            d2 = np.sum((cbir - cbjr)**2)
            if d2 <= cld2:
                return True, False
            if d2 <= ctd2:
                contact = True
                if mark0: ct0.add(ir)
                if mark1: ct1.add(jr)
    return clash, contact


@jit
def _mark_cont(ct0, ct1, bnd0, bnd1, cb0, cb1, mark0, mark1, cld2, ctd2):
    clash, contact = False, False
    for jr in range(*bnd1):
        cbjr = cb1[jr]
        for ir in range(*bnd0):
            cbir = cb0[ir]
            d2 = np.sum((cbir - cbjr)**2)
            if d2 <= cld2:
                return True, False
            if d2 <= ctd2:
                contact = True
                if mark0: ct0.add(ir)
                if mark1: ct1.add(jr)
    return clash, contact


@jit
def _clash_contact_by_helix(
        aln0, aln1, x, cb0, cb1, hrange0, hrange1, helixof0, helixof1, cld2,
        ctd2
):
    # at least two helices befor and after don't exist
    if (helixof0[aln0] < 2 or helixof1[aln1] < 0
            or helixof1[aln1] + 2 >= hrange1.shape[0]):
        return 0, 0, 0, 0

    ct0, ct1 = set([-1]), set([-1])

    helix_bounds_a = hrange0[helixof0[aln0] - 2]
    helix_bounds_b = hrange0[helixof0[aln0] - 1]
    helix_bounds_c = hrange0[helixof0[aln0] - 0]
    helix_bounds_d = hrange1[helixof1[aln1] + 0]
    helix_bounds_e = hrange1[helixof1[aln1] + 1]
    helix_bounds_f = hrange1[helixof1[aln1] + 2]
    helix_bounds_c[1] = aln0 + 1
    helix_bounds_d[0] = aln1

    ha = (helix_bounds_a[0], helix_bounds_a[1])
    hb = (helix_bounds_b[0], helix_bounds_b[1])
    hc = (helix_bounds_c[0], helix_bounds_c[1])
    hd = (helix_bounds_d[0], helix_bounds_d[1])
    he = (helix_bounds_e[0], helix_bounds_e[1])
    hf = (helix_bounds_f[0], helix_bounds_f[1])

    c0, t0 = _mark_cont_aln(ct0, ct1, ha, he, cb0, cb1, x, 1, 1, cld2, ctd2)
    if c0: return 1, 0, 0, 0
    c1, t1 = _mark_cont_aln(ct0, ct1, ha, hf, cb0, cb1, x, 1, 1, cld2, ctd2)
    if c1: return 1, 0, 0, 0
    c2, t2 = _mark_cont_aln(ct0, ct1, hb, he, cb0, cb1, x, 1, 1, cld2, ctd2)
    if c2: return 1, 0, 0, 0
    c3, t3 = _mark_cont_aln(ct0, ct1, hb, hf, cb0, cb1, x, 1, 1, cld2, ctd2)
    if c3: return 1, 0, 0, 0

    ncnh = len(ct0) + len(ct1) - 2  # -2 to remove the two -1's

    c4, t4 = _mark_cont_aln(ct0, ct1, hb, hd, cb0, cb1, x, 1, 0, cld2, ctd2)
    if c4: return 1, 0, 0, 0
    c5, t5 = _mark_cont_aln(ct0, ct1, ha, hd, cb0, cb1, x, 1, 0, cld2, ctd2)
    if c5: return 1, 0, 0, 0
    c6, t6 = _mark_cont_aln(ct0, ct1, hc, he, cb0, cb1, x, 0, 1, cld2, ctd2)
    if c6: return 1, 0, 0, 0
    c7, t7 = _mark_cont_aln(ct0, ct1, hc, hf, cb0, cb1, x, 0, 1, cld2, ctd2)
    if c7: return 1, 0, 0, 0

    c8, t8 = _mark_cont(ct0, ct0, ha, hc, cb0, cb0, 1, 0, cld2, ctd2)
    if c8: return 1, 0, 0, 0
    c9, t9 = _mark_cont(ct0, ct0, hb, hc, cb0, cb0, 1, 0, cld2, ctd2)
    if c9: return 1, 0, 0, 0
    ca, ta = _mark_cont(ct1, ct1, hd, he, cb1, cb1, 0, 1, cld2, ctd2)
    if ca: return 1, 0, 0, 0
    cb, tb = _mark_cont(ct1, ct1, hd, hf, cb1, cb1, 0, 1, cld2, ctd2)
    if cb: return 1, 0, 0, 0

    clash = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + ca + cb
    nc = len(ct0) + len(ct1) - 2

    nhc = 1 + (t4 or t9) + (t5 or t8) + (t6 or ta) + (t7 or tb)

    return clash, nc, ncnh, nhc


@jit
def _clash_contact_simple(
        aln0, aln1, xaln, ncac0, ncac1, splice_clash_contact_range, clashd2,
        contactd2
):
    nclash, ncontact = 0, 0
    for j in range(4, 3 * splice_clash_contact_range + 3, 3):
        if 3 * aln1 + j >= len(ncac1): continue
        b = xaln @ ncac1[3 * aln1 + j]
        for i in range(-2, -3 * splice_clash_contact_range - 1, -3):
            if 3 * aln0 + i < 0: continue
            a = ncac0[3 * aln0 + i]
            d2 = np.sum((a - b)**2)
            if d2 < clashd2:
                nclash += 1
            elif i % 3 == 1 and j % 3 == 1 and d2 < contactd2:
                ncontact += 1
    assert 0 <= np.isnan(nclash) < 99999, 'bad nclash'
    assert 0 <= np.isnan(ncontact) < 99999, 'bad ncontact'
    return nclash, ncontact, 999, 999


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
