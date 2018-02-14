import pytest
import os
from os.path import join, dirname, abspath, exists
try:
    import pyrosetta
    pyrosetta.init('-corrections:beta_nov16 -mute all')
    HAVE_PYROSETTA = True
    print("pyrosetta initialized successfully!")
except ImportError:
    print("no module pyrosetta")
    HAVE_PYROSETTA = False


@pytest.fixture(scope='session')
def pdbdir():
    root = join(dirname(__file__), '..')
    d = join(root, 'data')
    assert exists(d)
    return d


def get_pose(pdbdir, fname):
    if not HAVE_PYROSETTA:
        return None
    pose = pyrosetta.pose_from_file(join(pdbdir, fname))
    return pose


@pytest.fixture(scope='session')
def pose(pdbdir):
    return get_pose(pdbdir, 'small.pdb')


@pytest.fixture(scope='session')
def curved_helix_pose(pdbdir):
    return get_pose(pdbdir, 'curved_helix.pdb')


@pytest.fixture(scope='session')
def strand_pose(pdbdir):
    return get_pose(pdbdir, 'strand.pdb')


@pytest.fixture(scope='session')
def loop_pose(pdbdir):
    return get_pose(pdbdir, 'loop.pdb')


@pytest.fixture(scope='session')
def trimer_pose(pdbdir):
    return get_pose(pdbdir, '1coi.pdb')


@pytest.fixture(scope='session')
def trimerA_pose(pdbdir):
    return get_pose(pdbdir, '1coi_A.pdb')


@pytest.fixture(scope='session')
def trimerB_pose(pdbdir):
    return get_pose(pdbdir, '1coi_B.pdb')


@pytest.fixture(scope='session')
def trimerC_pose(pdbdir):
    return get_pose(pdbdir, '1coi_C.pdb')


@pytest.fixture(scope='session')
def c1pose(pdbdir):
    return get_pose(pdbdir, 'c1.pdb')


@pytest.fixture(scope='session')
def c2pose(pdbdir):
    return get_pose(pdbdir, 'c2.pdb')


@pytest.fixture(scope='session')
def c3pose(pdbdir):
    return get_pose(pdbdir, 'c3.pdb')


@pytest.fixture(scope='session')
def c3hetpose(pdbdir):
    return get_pose(pdbdir, 'c3het.pdb')


@pytest.fixture(scope='session')
def c3_splay_pose(pdbdir):
    return get_pose(pdbdir, 'c3_splay.pdb')


@pytest.fixture(scope='session')
def c4pose(pdbdir):
    return get_pose(pdbdir, 'c4.pdb')


@pytest.fixture(scope='session')
def c5pose(pdbdir):
    return get_pose(pdbdir, 'c5.pdb')


@pytest.fixture(scope='session')
def c6pose(pdbdir):
    return get_pose(pdbdir, 'c6.pdb')
