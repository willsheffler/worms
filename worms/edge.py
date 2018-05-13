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

    def __init__(self, splices):
        """TODO: Summary

        Args:
            splices (TYPE): Description
        """
        pass

    def allowed_splices(self, i):
        return self.splices[i, 2:self.splices[i, 1]]


_SCM_Scores = namedtuple('_SCM_Scores',
                         'aln0 aln1 n1b n2b nclash ncontact rms'.split())


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
                        aln0s,
                        bb1,
                        aln1s,
                        clashd2=3.0**2,
                        rms_range=9,
                        clash_contact_range=9):
    scm = np.ones((len(aln0s), len(aln1s)), dtype=np.float32)
    df = _SCM_Scores(
        aln0=np.array(aln0s, dtype=np.int32),
        aln1=np.array(aln1s, dtype=np.int32),
        n1b=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        n2b=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        nclash=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        ncontact=np.zeros((len(aln0s), len(aln1s)), dtype=np.int32),
        rms=np.zeros((len(aln0s), len(aln1s)), dtype=np.float32),
    )
    for ialn0, aln0 in enumerate(aln0s):
        chainb0 = chainbounds_of_ires(bb0, aln0)
        stub0 = bb0.stubs[aln0]
        for ialn1, aln1 in enumerate(aln1s):
            chainb1 = chainbounds_of_ires(bb1, aln1)
            stub1 = bb1.stubs[aln1]
            xaln = stub0 @ np.linalg.inv(stub1)

            for i in range(-1, -clash_contact_range - 1, -1):
                if (aln0 + i < chainb0[0]): break
                for j in range(1, clash_contact_range + 1):
                    if ((aln0 + j >= len(bb0.ncac))
                            or (aln1 + j >= len(bb1.ncac))):
                        break
                    for ia in range(3):
                        for ja in range(3):
                            a = bb0.ncac[aln0 + i, ia]
                            b = xaln @ bb1.ncac[aln1 + j, ja]
                            d2 = np.sum((a - b)**2)
                            df.n2b[ialn0, ialn1] += 1
                            if d2 < clashd2:
                                df.nclash[ialn0, ialn1] += 1
                            elif ia == 1 and ja == 1 and d2 < 64:
                                df.ncontact[ialn0, ialn1] += 1

            for i in range(-1, -rms_range - 1, -1):
                if (aln0 + i < chainb0[0]) or (aln1 + i < chainb1[0]):
                    break
                for ia in range(3):
                    a = bb0.ncac[aln0 + i, ia]
                    b = xaln @ bb1.ncac[aln1 + i, ia]
                    df.rms[ialn0, ialn1] += np.sum((a - b)**2)
                    df.n1b[ialn0, ialn1] += 1
            for j in range(0, rms_range + 1):
                if (aln0 + j >= chainb0[1]) or (aln1 + j >= chainb1[1]):
                    break
                for ja in range(3):
                    a = bb0.ncac[aln0 + j, ja]
                    b = xaln @ bb1.ncac[aln1 + j, ja]
                    df.rms[ialn0, ialn1] += np.sum((a - b)**2)
                    df.n1b[ialn0, ialn1] += 1

            df.rms[ialn0, ialn1] = np.sqrt(
                df.rms[ialn0, ialn1] / df.n1b[ialn0, ialn1])

    return df


def splice_metrics(u, ubbs, v, vbbs):
    assert (u.dirn[1] + v.dirn[0]) == 1
    outidx = [
        np.where(u.inout[:, 1] == i)[0][0]
        for i in range(np.max(u.inout[:, 1]) + 1)
    ]

    outbb = u.ibb[outidx]
    outres = u.ires[outidx, 1]
    inbb = v.ibb[v.inbreaks[:-1]]
    inres = v.ires[v.inbreaks[:-1], 0]
    outbb_breaks = contig_idx_breaks(outbb)
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
            if u.dirn[1] == 1:  # 'CN'
                part = _jit_splice_metrics(bb0, ires0, bb1, ires1)
            else:
                part = _jit_splice_metrics(bb1, ires1, bb0, ires0)
                part = _SCM_Scores(*(p.T for p in part))
            cols.append(part)
        scm.append(scm_concat(cols, axis=1))
    scm = scm_concat(scm, axis=0)

    print(scm[0].shape, np.sum(scm[0]))
    right_shape = (len(outidx), len(v.inbreaks) - 1)
    assert all(x.shape == right_shape for x in scm if x.ndim == 2)

    return scm


def Edge(u, ubbs, v, vbbs):
    return _Edge()