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

_SCM_Scores = namedtuple('_SCM_Scores', 'nclash ncontact rms'.split())


def scm_concat(lst, axis=0):
    result = list()
    for fieldn in _SCM_Scores._fields:
        ndim = getattr(lst[0], fieldn).ndim
        tmpaxis = -1 if ndim is 1 else axis
        result.append(
            np.concatenate([getattr(x, fieldn) for x in lst], axis=tmpaxis)
        )
    return _SCM_Scores(*result)


@jit
def _chainbounds_of_ires(chains, ires):
    for c in range(len(chains)):
        if chains[c, 0] <= ires < chains[c, 1]:
            return chains[c, 0], chains[c, 1]
    return (-1, -1)


@jit
def _jit_splice_metrics(chains0, chains1,
                        ncac0_3d, ncac1_3d,
                        stubs0, stubs1,
                        aln0s, aln1s,
                        clashd2=3.0**2,
                        contactd2=10.0**2,
                        rms_range=9,
                        clash_contact_range=9,
                        rms_cut=1.1,
                        skip_on_fail=True):  # yapf: disable

    out_rms = np.zeros((len(aln0s), len(aln1s)), dtype=np.float32)
    out_nclash = -np.ones((len(aln0s), len(aln1s)), dtype=np.float32)
    out_ncontact = -np.ones((len(aln0s), len(aln1s)), dtype=np.float32)

    ncac0 = ncac0_3d.reshape(-1, 4)
    ncac1 = ncac1_3d.reshape(-1, 4)

    b = np.empty((4, ), dtype=np.float64)

    for ialn1, aln1 in enumerate(aln1s):
        chainb10, chainb11 = _chainbounds_of_ires(chains1, aln1)
        if np.abs(chainb10 - aln1) < rms_range: continue
        if np.abs(chainb11 - aln1) <= rms_range: continue
        stub1_inv = np.linalg.inv(stubs1[aln1])

        for ialn0, aln0 in enumerate(aln0s):
            chainb00, chainb01 = _chainbounds_of_ires(chains0, aln0)
            if np.abs(chainb00 - aln0) < rms_range: continue
            if np.abs(chainb01 - aln0) <= rms_range: continue
            xaln = stubs0[aln0] @ stub1_inv

            sum_d2, n1b = 0.0, 0
            for i in range(-3 * rms_range, 3 * rms_range + 3):
                a = ncac0[3 * aln0 + i]
                b[:] = xaln @ ncac1[3 * aln1 + i]
                sum_d2 += np.sum((a - b)**2)
            rms = np.sqrt(sum_d2 / (rms_range * 6 + 3))
            assert 0 <= rms < 9e9, 'bad rms'
            out_rms[ialn0, ialn1] = rms

            if skip_on_fail and rms > rms_cut:
                continue

            nclash, ncontact = 0, 0
            for j in range(3, 3 * clash_contact_range + 3):
                b[:] = xaln @ ncac1[3 * aln1 + j]
                for i in range(-1, -3 * clash_contact_range - 1, -1):
                    a = ncac0[3 * aln0 + i]
                    d2 = np.sum((a - b)**2)
                    if d2 < clashd2:
                        nclash += 1
                    elif i % 3 == 1 and j % 3 == 1 and d2 < contactd2:
                        ncontact += 1
            assert 0 <= np.isnan(nclash) < 99999, 'bad nclash'
            assert 0 <= np.isnan(ncontact) < 99999, 'bad ncontact'
            out_nclash[ialn0, ialn1] = nclash
            out_ncontact[ialn0, ialn1] = ncontact

    return out_rms, out_nclash, out_ncontact


def get_allowed_splices(
        u,
        ublks,
        v,
        vblks,
        splicedb=None,
        rms_cut=0.7,
        ncontact_cut=10,
        clashd2=3.0**2,
        contactd2=10.0**2,
        rms_range=9,
        clash_contact_range=9,
        skip_on_fail=True,
        parallel=False,
        progressbar=False,
        verbosity=1
):
    if splicedb is None: splicedb = SpliceDB()
    params = (
        rms_cut, ncontact_cut, clashd2, contactd2, rms_range,
        clash_contact_range
    )

    assert (u.dirn[1] + v.dirn[0]) == 1, 'get_allowed_splices dirn mismatch'
    outidx = [
        np.where(u.inout[:, 1] == i)[0][0]
        for i in range(np.max(u.inout[:, 1]) + 1)
    ]

    outblk = u.ibblock[outidx]
    outres = u.ires[outidx, 1]
    inblk = v.ibblock[v.inbreaks[:-1]]
    inres = v.ires[v.inbreaks[:-1], 0]
    inblk_breaks = contig_idx_breaks(inblk)

    outblk_res = defaultdict(list)
    for iblk, ires in zip(outblk, outres):
        outblk_res[iblk].append(ires)

    inblk_res = defaultdict(list)
    for iblk, ires in zip(inblk, inres):
        inblk_res[iblk].append(ires)

    for iblk in inblk_res.keys():
        inblk_res[iblk] = np.array(inblk_res[iblk], 'i4')
    for iblk in outblk_res.keys():
        outblk_res[iblk] = np.array(outblk_res[iblk], 'i4')

    Nout = sum(len(a) for a in outblk_res.values())
    valid_splices = [list() for i in range(Nout)]

    swapped = False
    if u.dirn[1] == 0:  # swap so N-to-C!
        swapped = True
        u, ublks, v, vblks = v, vblks, u, ublks
        outblk_res, inblk_res = inblk_res, outblk_res
        outblk, inblk = inblk, outblk

    exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        ofst0 = 0
        for iblk0, ires0 in outblk_res.items():
            blk0 = ublks[iblk0]
            pdb0 = bytes(blk0.file)
            c_side_cache = splicedb.c_side_cache(params, pdb0)
            ofst1 = 0
            for iblk1, ires1 in inblk_res.items():
                blk1 = vblks[iblk1]
                pdb1 = bytes(blk1.file)
                if pdb1 in c_side_cache:
                    future = NonFuture(c_side_cache[pdb1][swapped])
                else:
                    future = pool.submit(
                        _jit_splice_metrics, blk0.chains, blk1.chains,
                        blk0.ncac, blk1.ncac, blk0.stubs, blk1.stubs, ires0,
                        ires1, clashd2, contactd2, rms_range,
                        clash_contact_range, rms_cut, skip_on_fail
                    )
                fs = (iblk0, iblk1, ofst0, ofst1, len(ires0), len(ires1))
                future.stash = fs
                futures.append(future)
                ofst1 += len(ires1)
            ofst0 += len(ires0)

        future_iter = cf.as_completed(futures)
        if progressbar:
            future_iter = tqdm(cf.as_completed(futures), total=len(futures))

        for future in future_iter:
            iblk0, iblk1, ofst0, ofst1, nres0, nres1 = future.stash
            result = future.result()
            if len(result) is 3 and isinstance(result[0], np.ndarray):
                rms, nclash, ncontact = result
                good_edges = ((nclash == 0) * (rms <= rms_cut) *
                              (ncontact >= ncontact_cut))
                assert good_edges.shape == (nres0, nres1)
                if swapped: good_edges = good_edges.T

                result, result_swp = list(), list()
                for a in good_edges:
                    good = np.nonzero(a)[0]
                    result.append(good if len(good) else None)
                for a in good_edges.T:
                    good = np.nonzero(a)[0]
                    result_swp.append(good if len(good) else None)

                # store in SpliceDB
                pdb0 = bytes(ublks[iblk0].file)  # C-term side
                pdb1 = bytes(vblks[iblk1].file)  # N-term side
                splicedb.add_to_c_side_cache(
                    params, pdb0, pdb1,
                    (result_swp, result) if swapped else (result, result_swp)
                )

            if swapped:
                assert len(result) == nres1
                ofst0, ofst1 = ofst1, ofst0
            else:
                assert len(result) == nres0

            for i, a in enumerate(result):
                if a is not None:
                    valid_splices[i + ofst0].extend(a + ofst1)

    return valid_splices


def Edge(u, ublks, v, vblks, **kw):

    m = splice_metrics(u, ublks, v, vblks, **kw)
    # * is logical 'and'
    good_edges = ((m.nclash == 0) * (m.rms <= 0.7) * (m.ncontact >= 10))
    if np.sum(good_edges) == 0:
        raise ValueError('all splices invalid ')
    print(
        'fraction good edges:', good_edges.sum(), good_edges.size,
        np.round(good_edges.sum() / good_edges.size, 4)
    )
    splices = scmatrix_to_splices(good_edges)

    valid_splices = get_allowed_splices(u, ublks, v, vblks, **kw)
    maxentries = max(len(_) for _ in valid_splices)
    splice_ary = np.zeros((len(valid_splices), maxentries + 1),
                          dtype=np.int32) - 1
    for i, a in enumerate(splice_ary):
        a[0] = len(valid_splices[i]) + 1
        a[1:a[0]] = sorted(valid_splices[i])

    assert np.max(splice_ary[:, 1:]
                  ) < len(v.inbreaks), 'egde.py bad splice_ary'
    assert len(splice_ary) == 1 + np.max(u.inout[:, 1]
                                         ), 'edge.py, bad splice_ary'

    assert len(splices) == len(splice_ary)
    assert np.all(splices == splice_ary)
    print('pass')

    return _Edge(splice_ary)


def splice_metrics(
        u,
        ublks,
        v,
        vblks,
        splicedb=None,
        clashd2=3.0**2,
        contactd2=10.0**2,
        rms_range=9,
        clash_contact_range=9,
        rms_cut=1.1,
        skip_on_fail=True,
        parallel=False,
        progressbar=False,
        verbosity=0
):

    assert (u.dirn[1] + v.dirn[0]) == 1
    outidx = [
        np.where(u.inout[:, 1] == i)[0][0]
        for i in range(np.max(u.inout[:, 1]) + 1)
    ]

    outblk = u.ibblock[outidx]
    outres = u.ires[outidx, 1]
    inblk = v.ibblock[v.inbreaks[:-1]]
    inres = v.ires[v.inbreaks[:-1], 0]
    # outblk_breaks = contig_idx_breaks(outblk)
    inblk_breaks = contig_idx_breaks(inblk)

    outblk_res = defaultdict(list)
    for iblk, ires in zip(outblk, outres):
        outblk_res[iblk].append(ires)

    inblk_res = defaultdict(list)
    for iblk, ires in zip(inblk, inres):
        inblk_res[iblk].append(ires)

    for iblk in inblk_res.keys():
        inblk_res[iblk] = np.array(inblk_res[iblk], 'i4')
    for iblk in outblk_res.keys():
        outblk_res[iblk] = np.array(outblk_res[iblk], 'i4')

    swapped = False
    if u.dirn[1] == 0:  # swap!
        swapped = True
        u, ublks, v, vblks = v, vblks, u, ublks
        outblk_res, inblk_res = inblk_res, outblk_res
        outblk, inblk = inblk, outblk

    metrics = _SCM_Scores(
        nclash=np.zeros((len(outblk), len(inblk)), dtype=np.int32) - 1,
        ncontact=np.zeros((len(outblk), len(inblk)), dtype=np.int32) - 1,
        rms=np.zeros((len(outblk), len(inblk)), dtype=np.float32) - 1
    )

    exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        offset0 = 0
        for iblk0, ires0 in outblk_res.items():
            blk0 = ublks[iblk0]
            offset1 = 0
            for iblk1, ires1 in inblk_res.items():
                blk1 = vblks[iblk1]
                future = pool.submit(
                    _jit_splice_metrics, blk0.chains, blk1.chains, blk0.ncac,
                    blk1.ncac, blk0.stubs, blk1.stubs, ires0, ires1, clashd2,
                    contactd2, rms_range, clash_contact_range, rms_cut,
                    skip_on_fail
                )
                future.stash = (
                    iblk0, iblk1, offset0, offset1, len(ires0), len(ires1)
                )
                futures.append(future)
                offset1 += len(ires1)
            offset0 += len(ires0)

        iter = cf.as_completed(futures)
        if progressbar:
            iter = tqdm(cf.as_completed(futures), total=len(futures))
        for i, future in enumerate(iter):
            iblk0, iblk1, offset0, offset1, nres0, nres1 = future.stash
            rms, nclash, ncontact = future.result()
            myslice = (
                slice(offset0, offset0 + nres0),
                slice(offset1, offset1 + nres1)
            )
            metrics.rms[myslice] = rms
            metrics.nclash[myslice] = nclash
            metrics.ncontact[myslice] = ncontact

            # good_edges = ((nclash == 0) * (rms <= rms_cut) *
            #               (ncontact >= ncontact_cut))
            # assert good_edges.shape == (nres0, nres1)
            # if swapped: good_edges = good_edges.T
            # result, result_swp = list(), list()
            # for a in good_edges:
            #     good = np.nonzero(a)[0]
            #     result.append(good if len(good) else None)
            # for a in good_edges.T:
            #     good = np.nonzero(a)[0]
            #     result_swp.append(good if len(good) else None)

    if swapped:  # unswap!
        metrics = _SCM_Scores(
            metrics.nclash.T, metrics.ncontact.T, metrics.rms.T
        )

    return metrics


@jit
def scmatrix_to_splices(scmatrix):
    assert scmatrix.ndim is 2, 'bad scmatrix'
    nout = scmatrix.shape[0]
    max_Nin = 0
    for i in range(scmatrix.shape[0]):
        max_Nin = max(max_Nin, np.sum(scmatrix[i]))
    splices = np.zeros((nout, max_Nin + 1), dtype=np.int32)
    splices -= 1
    for i in range(nout):
        non0 = scmatrix[i].nonzero()[0].astype(np.int32)
        splices[i, 0] = len(non0) + 1
        splices[i, 1:len(non0) + 1] = non0
    assert splices.shape[0] == scmatrix.shape[0], 'bad splices'
    return splices


@nb.jitclass((
    ('splices', nt.int32[:, :]),
))  # yapf: disable
class _Edge:
    """contains junction scores
    """

    def __init__(self, splices):
        self.splices = splices

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
        return (self.splices, )