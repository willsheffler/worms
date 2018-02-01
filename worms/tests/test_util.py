from .. import util
import pytest
try:
    import pyrosetta
    HAVE_PYROSETTA = True
except ImportError:
    HAVE_PYROSETTA = False


@pytest.mark.xfail()
def test_infer_symmetry(c1pose, c2pose, c3pose, c3hetpose, c6pose):
    print(c3pose)

    assert 0
