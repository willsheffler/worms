import numpy as np
import numba as nb
import numba.types as nt
from collections import defaultdict
from worms.util import contig_idx_breaks

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


@nb.njit(nogil=1)
def splice_compatibility_matrix(bb0, ires0, bb1, ires1):
    scm = np.empty((len(ires0), len(ires1)), dtype=nt.float32)
    return scm


# Vertex fields
#     ('x2exit', nt.float32[:, :, :]),
#     ('x2orig', nt.float32[:, :, :]),
#     ('inout' , nt.int32[:, :]),
#     ('inbreaks' , nt.int32[:]),
#     ('ires'  , nt.int32[:, :]),
#     ('isite' , nt.int32[:, :]),
#     ('ichain' , nt.int32[:, :]),
#     ('ibb' , nt.int32[:]),
#     ('dirn'   , nt.int32[:]),
def Edge(u, ubbs, v, vbbs):
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

    rows = list()
    for ibb0, ires0 in outbb_res.items():
        bb0 = ubbs[ibb0]
        cols = list()
        for ibb1, ires1 in inbb_res.items():
            bb1 = vbbs[ibb1]
            if u.dirn[1] == 0:  # 'CN'
                part = splice_compatibility_matrix(bb0, ires0, bb1, ires1)
            else:
                part = splice_compatibility_matrix(bb1, ires1, bb0, ires0).T
            cols.append(part)
        rows.append(np.concatenate(cols, axis=1))
    scm = np.concatenate(rows, axis=0)

    print(scm.shape)
    assert scm.shape == (len(outidx), len(v.inbreaks) - 1)

    return _Edge(scm)
