import numpy as np
import numba as nb
import numba.types as nt
from collections import defaultdict, namedtuple
from worms.util import contig_idx_breaks
from worms.bblock import chainbounds_of_ires

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

    def allowed_splices(self, i):
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


@nb.njit(nogil=1)
def _jit_splice_metrics(bb0,
                        bb1,
                        ncac0,
                        ncac1,
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
    scm = np.ones((len(aln0s), len(aln1s)), dtype=np.float32)
    df = _SCM_Scores(
        # aln0=np.array(aln0s, dtype=np.int32),
        # aln1=np.array(aln1s, dtype=np.int32),
        # n1b=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        # n2b=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        nclash=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        ncontact=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        rms=np.ones((len(aln0s), len(aln1s)), dtype=np.float32) * 9e9,
    )
    for ialn0, aln0 in enumerate(aln0s):
        chainb0 = chainbounds_of_ires(bb0, aln0)
        if np.abs(chainb0[0] - aln0) < rms_range: continue
        if np.abs(chainb0[1] - aln0) < rms_range: continue
        # print('chnb0', 10000 + ialn0, 10000 + aln0, 10000 + int(chainb0[0]),
        # 10000 + int(chainb0[1]))
        stub0 = stubs0[aln0]
        for ialn1, aln1 in enumerate(aln1s):
            chainb1 = chainbounds_of_ires(bb1, aln1)
            if np.abs(chainb1[0] - aln1) < rms_range: continue
            if np.abs(chainb1[1] - aln1) < rms_range: continue
            # print('chnb1', 10000 + ialn1, 10000 + aln1,
            # 10000 + int(chainb1[0]), 10000 + int(chainb1[1]))
            stub1 = stubs1[aln1]
            xaln = stub0 @ np.linalg.inv(stub1)

            rms, n1b = 0.0, 0
            for i in range(-3 * rms_range, 3 * rms_range + 3):
                a = ncac0[3 * aln0 + i]
                b = xaln @ ncac1[3 * aln1 + i]
                rms += np.sum((a - b)**2)
            rms = np.sqrt(rms / (rms_range * 6 + 3))
            df.rms[ialn0, ialn1] = rms

            if skip_on_fail and rms > rms_cut:
                continue

            nclash, ncontact = 0, 0
            for j in range(3, 3 * clash_contact_range + 3):
                b = xaln @ ncac1[3 * aln1 + j]
                for i in range(-1, -3 * clash_contact_range - 1, -1):
                    a = ncac0[3 * aln0 + i]
                    d2 = np.sum((a - b)**2)
                    if d2 < clashd2:
                        nclash += 1
                    elif i % 3 == 1 and j % 3 == 1 and d2 < contactd2:
                        ncontact += 1
            df.nclash[ialn0, ialn1] = nclash
            df.ncontact[ialn0, ialn1] = ncontact
    return df


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

    scm = list()
    for ibb0, ires0 in outbb_res.items():
        bb0 = ubbs[ibb0]
        cols = list()
        for ibb1, ires1 in inbb_res.items():
            bb1 = vbbs[ibb1]
            if u.dirn[1] == 1:  # '*C'->'N*'
                assert v.dirn[0] == 0
                part = _jit_splice_metrics(bb0, bb1, bb0.ncac.reshape(-1, 4),
                                           bb1.ncac.reshape(-1, 4), bb0.stubs,
                                           bb1.stubs, ires0, ires1, **kw)
            else:
                assert v.dirn[0] == 1
                part = _jit_splice_metrics(bb1, bb0, bb1.ncac.reshape(-1, 4),
                                           bb0.ncac.reshape(-1, 4), bb1.stubs,
                                           bb0.stubs, ires1, ires0, **kw)
                part = _SCM_Scores(*(p.T for p in part))
            cols.append(part)
        scm.append(scm_concat(cols, axis=1))
    scm = scm_concat(scm, axis=0)

    right_shape = (len(outidx), len(v.inbreaks) - 1)
    assert all(x.shape == right_shape for x in scm if x.ndim == 2)

    return scm


def Edge(u, ubbs, v, vbbs, rms_cut=1.1, ncontact_cut=10, **kw):

    m = splice_metrics(u, ubbs, v, vbbs, rms_cut=rms_cut, **kw)

    # * is logical 'and'
    good_edges = ((m.nclash == 0) * (m.rms <= rms_cut) *
                  (m.ncontact >= ncontact_cut))
    return _Edge(good_edges)