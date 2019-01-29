import pytest
import os
import sys
from os.path import join, dirname, abspath, exists

sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from worms.database import CachingBBlockDB, CachingSpliceDB

try:
    import pyrosetta

    pyrosetta.init(
        "-corrections:beta_nov16 -mute all -preserve_crystinfo -symmetry::symmetry_definition dummy"
    )
    HAVE_PYROSETTA = True
    print("pyrosetta initialized successfully!")
except ImportError:
    print("no module pyrosetta")
    HAVE_PYROSETTA = False


@pytest.fixture(scope="session")
def bbdb(datadir):
    return CachingBBlockDB(
        cachedirs=[str(".worms_pytest_cache")],
        dbfiles=[os.path.join(datadir, "test_db_file.json")],
        lazy=False,
        read_new_pdbs=HAVE_PYROSETTA,
    )


@pytest.fixture(scope="session")
def spdb(datadir):
    return CachingSpliceDB(cachedirs=[str(".worms_pytest_cache")])


@pytest.fixture(scope="session")
def bbdb_fullsize_prots(datadir):
    return CachingBBlockDB(
        cachedirs=[str(".worms_pytest_cache")],
        dbfiles=[os.path.join(datadir, "test_fullsize_prots.json")],
        lazy=False,
        read_new_pdbs=HAVE_PYROSETTA,
    )


@pytest.fixture(scope="session")
def pdbdir():
    root = join(dirname(__file__), "..")
    d = join(root, "data")
    assert exists(d)
    return d


@pytest.fixture(scope="session")
def datadir():
    root = join(dirname(__file__), "..")
    d = join(root, "data")
    assert exists(d)
    return d


def get_pose(pdbdir, fname):
    if not HAVE_PYROSETTA:
        return None
    return pyrosetta.pose_from_file(join(pdbdir, fname))
    # return tmp.pose(join(pdbdir, fname))


@pytest.fixture(scope="session")
def pose(pdbdir):
    return get_pose(pdbdir, "small.pdb")


@pytest.fixture(scope="session")
def curved_helix_pose(pdbdir):
    return get_pose(pdbdir, "curved_helix.pdb")


@pytest.fixture(scope="session")
def strand_pose(pdbdir):
    return get_pose(pdbdir, "strand.pdb")


@pytest.fixture(scope="session")
def loop_pose(pdbdir):
    return get_pose(pdbdir, "loop.pdb")


@pytest.fixture(scope="session")
def trimer_pose(pdbdir):
    return get_pose(pdbdir, "1coi.pdb")


@pytest.fixture(scope="session")
def trimerA_pose(pdbdir):
    return get_pose(pdbdir, "1coi_A.pdb")


@pytest.fixture(scope="session")
def trimerB_pose(pdbdir):
    return get_pose(pdbdir, "1coi_B.pdb")


@pytest.fixture(scope="session")
def trimerC_pose(pdbdir):
    return get_pose(pdbdir, "1coi_C.pdb")


@pytest.fixture(scope="session")
def c1pose(pdbdir):
    return get_pose(pdbdir, "c1.pdb")


@pytest.fixture(scope="session")
def c2pose(pdbdir):
    return get_pose(pdbdir, "c2.pdb")


@pytest.fixture(scope="session")
def c3pose(pdbdir):
    return get_pose(pdbdir, "c3.pdb")


@pytest.fixture(scope="session")
def c3hetpose(pdbdir):
    return get_pose(pdbdir, "c3het.pdb")


@pytest.fixture(scope="session")
def c3_splay_pose(pdbdir):
    return get_pose(pdbdir, "c3_splay.pdb")


@pytest.fixture(scope="session")
def c4pose(pdbdir):
    return get_pose(pdbdir, "c4.pdb")


@pytest.fixture(scope="session")
def c5pose(pdbdir):
    return get_pose(pdbdir, "c5.pdb")


@pytest.fixture(scope="session")
def c6pose(pdbdir):
    return get_pose(pdbdir, "c6.pdb")


@pytest.fixture(scope="session")
def db_asu_pose(pdbdir):
    return get_pose(pdbdir, "db_asu.pdb")


@pytest.fixture(scope="session")
def hetC2A_pose(pdbdir):
    return get_pose(pdbdir, "hetC2A.pdb")


@pytest.fixture(scope="session")
def hetC2B_pose(pdbdir):
    return get_pose(pdbdir, "hetC2B.pdb")
