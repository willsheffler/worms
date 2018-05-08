import numpy as np
import numba as nb
import numba.types as nt

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


def Edge(vert_in, vert_out):
    # ???
    return _Edge(np.eye(2))
