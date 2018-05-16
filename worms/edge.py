import numpy as np
import numba as nb
import numba.types as nt
from collections import defaultdict, namedtuple
from worms.util import contig_idx_breaks, jit

try:
    # this is such bullshit...
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
    HAVE_PYROSETTA = True
except ImportError:
    HAVE_PYROSETTA = False



@nb.jitclass((
    ('splices', nt.int32[:, :]),
))  # yapf: disable
class _Edge:
    """contains junction scores
    """

    def __init__(self, scmatrix):
        """TODO: Summary

        Args:
            splices (TYPE): Description
        """
        assert scmatrix.ndim is 2
        nout = scmatrix.shape[0]
        max_Nin = 0
        for i in range(scmatrix.shape[0]):
            max_Nin = max(max_Nin, np.sum(scmatrix[i]))

        self.splices = np.zeros((nout, max_Nin + 1), dtype=np.int32)
        self.splices -= 1
        for i in range(nout):
            non0 = scmatrix[i].nonzero()[0].astype(np.int32)
            self.splices[i, 0] = len(non0) + 1
            self.splices[i, 1:len(non0) + 1] = non0

    @property
    def len(self):
        return len(self.splices)

    def allowed_entries(self, i):
        return self.splices[i, 1:self.splices[i, 0]]


_SCM_Scores = namedtuple('_SCM_Scores', 'nclash ncontact rms'.split())


def scm_concat(lst, axis=0):
    result = list()
    for fieldn in _SCM_Scores._fields:
        ndim = getattr(lst[0], fieldn).ndim
        tmpaxis = -1 if ndim is 1 else axis
        result.append(
            np.concatenate([getattr(x, fieldn) for x in lst], axis=tmpaxis))
    return _SCM_Scores(*result)


@jit
def _chainbounds_of_ires(chains, ires):
    for c in range(len(chains)):
        if chains[c, 0] <= ires < chains[c, 1]:
            return chains[c, 0], chains[c, 1]
    return (-1, -1)


@jit
def _jit_splice_metrics(out_offset0,
                        out_offset1,
                        out_rms,
                        out_nclash,
                        out_ncontact,
                        chains0,
                        chains1,
                        ncac0_3d,
                        ncac1_3d,
                        stubs0,
                        stubs1,
                        aln0s,
                        aln1s,
                        clashd2=3.0**2,
                        contactd2=10.0**2,
                        rms_range=9,
                        clash_contact_range=9,
                        rms_cut=1.1,
                        skip_on_fail=True):

    assert out_nclash.shape == out_rms.shape
    assert out_nclash.shape == out_ncontact.shape
    assert out_ncontact.shape == out_rms.shape
    assert out_offset0 + len(aln0s) <= out_rms.shape[0]
    assert out_offset1 + len(aln1s) <= out_rms.shape[1]

    ncac0 = ncac0_3d.reshape(-1, 4)
    ncac1 = ncac1_3d.reshape(-1, 4)

    b = np.empty((4, ), dtype=np.float64)

    for ialn1, aln1 in enumerate(aln1s):
        chainb10, chainb11 = _chainbounds_of_ires(chains1, aln1)
        if np.abs(chainb10 - aln1) < rms_range: continue
        if np.abs(chainb11 - aln1) < rms_range: continue
        iout1 = out_offset1 + ialn1
        stub1_inv = np.linalg.inv(stubs1[aln1])

        for ialn0, aln0 in enumerate(aln0s):
            chainb00, chainb01 = _chainbounds_of_ires(chains0, aln0)
            if np.abs(chainb00 - aln0) < rms_range: continue
            if np.abs(chainb01 - aln0) < rms_range: continue
            iout0 = out_offset0 + ialn0
            xaln = stubs0[aln0] @ stub1_inv

            sum_d2, n1b = 0.0, 0
            for i in range(-3 * rms_range, 3 * rms_range + 3):
                a = ncac0[3 * aln0 + i]
                b[:] = xaln @ ncac1[3 * aln1 + i]
                sum_d2 += np.sum((a - b)**2)
            rms = np.sqrt(sum_d2 / (rms_range * 6 + 3))
            out_rms[iout0, iout1] = rms

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
            out_nclash[iout0, iout1] = nclash
            out_ncontact[iout0, iout1] = ncontact


def splice_metrics(u, ubbs, v, vbbs, **kw):
    assert (u.dirn[1] + v.dirn[0]) == 1
    outidx = [
        np.where(u.inout[:, 1] == i)[0][0]
        for i in range(np.max(u.inout[:, 1]) + 1)
    ]

    outbb = u.ibb[outidx]
    outres = u.ires[outidx, 1]
    inbb = v.ibb[v.inbreaks[:-1]]
    inres = v.ires[v.inbreaks[:-1], 0]
    # outbb_breaks = contig_idx_breaks(outbb)
    inbb_breaks = contig_idx_breaks(inbb)

    outbb_res = defaultdict(list)
    for ibb, ires in zip(outbb, outres):
        outbb_res[ibb].append(ires)

    inbb_res = defaultdict(list)
    for ibb, ires in zip(inbb, inres):
        inbb_res[ibb].append(ires)

    for ibb in inbb_res.keys():
        inbb_res[ibb] = np.array(inbb_res[ibb], 'i4')
    for ibb in outbb_res.keys():
        outbb_res[ibb] = np.array(outbb_res[ibb], 'i4')

    if u.dirn[1] == 0:  # swap!
        u, ubbs, v, vbbs = v, vbbs, u, ubbs
        outbb_res, inbb_res = inbb_res, outbb_res
        outbb, inbb = inbb, outbb

    metrics = _SCM_Scores(
        nclash=np.zeros((len(outbb), len(inbb)), dtype=np.int32) - 1,
        ncontact=np.zeros((len(outbb), len(inbb)), dtype=np.int32) - 1,
        rms=np.zeros((len(outbb), len(inbb)), dtype=np.float32) - 1)

    offset0 = 0
    for ibb0, ires0 in outbb_res.items():
        bb0 = ubbs[ibb0]
        offset1 = 0
        for ibb1, ires1 in inbb_res.items():
            bb1 = vbbs[ibb1]
            _jit_splice_metrics(
                offset0,
                offset1,
                metrics.rms,
                metrics.nclash,
                metrics.ncontact,
                bb0.chains,
                bb1.chains,
                bb0.ncac,
                bb1.ncac,
                bb0.stubs,
                bb1.stubs,
                ires0,
                ires1,
                **kw,
            )
            offset1 += len(ires1)
        offset0 += len(ires0)

    if u.dirn[1] == 0:  # swap!
        metrics = _SCM_Scores(metrics.nclash.T, metrics.ncontact.T,
                              metrics.rms.T)

    return metrics


def Edge(u, ubbs, v, vbbs, rms_cut=1.1, ncontact_cut=10, **kw):

    m = splice_metrics(u, ubbs, v, vbbs, rms_cut=rms_cut, **kw)

    # * is logical 'and'
    good_edges = ((m.nclash == 0) * (m.rms <= rms_cut) *
                  (m.ncontact >= ncontact_cut))
    return _Edge(good_edges)