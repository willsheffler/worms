import pytest
import os

from os.path import join, dirname, abspath, exists
from deferred_import import deferred_import
import worms

# is there a better way?
# import sys
# sys.path.insert(0, os.path.dirname(__file__) + "/../..")

fixture = pytest.fixture(scope="session")

@fixture
def pdbdir():
   return worms.data.pdb_dir

@fixture
def datadir():
   return worms.data.data_dir

@fixture
def jsondir():
   return worms.data.json_dir

@fixture
def db_bblock_caching_v0(datadir):
   return worms.data.db_bblock_caching_v0(datadir)

@fixture
def db_splice_caching_v0(datadir):
   return worms.database.CachingSpliceDB(cachedirs=[str(".worms_pytest_cache")])

@fixture
def db_caching_bblock_v0_fullsize_prots(datadir):
   return worms.database.CachingBBlockDB(
      cachedirs=[str(".worms_pytest_cache")],
      dbfiles=[os.path.join(datadir, 'databases', 'json', 'test_fullsize_prots.json')],
      lazy=False,
      read_new_pdbs=worms.tests.HAVE_PYROSETTA,
   )

######################## pose stuff ###############################

def get_pose(pdbdir, fname):
   return worms.rosetta_init.pose_from_file(join(pdbdir, fname))
   # return tmp.pose(join(pdbdir, fname))

@fixture
def pose(pdbdir):
   return get_pose(pdbdir, "small.pdb")

@fixture
def curved_helix_pose(pdbdir):
   return get_pose(pdbdir, "curved_helix.pdb")

@fixture
def strand_pose(pdbdir):
   return get_pose(pdbdir, "strand.pdb")

@fixture
def loop_pose(pdbdir):
   return get_pose(pdbdir, "loop.pdb")

@fixture
def trimer_pose(pdbdir):
   return get_pose(pdbdir, "1coi.pdb")

@fixture
def trimerA_pose(pdbdir):
   return get_pose(pdbdir, "1coi_A.pdb")

@fixture
def trimerB_pose(pdbdir):
   return get_pose(pdbdir, "1coi_B.pdb")

@fixture
def trimerC_pose(pdbdir):
   return get_pose(pdbdir, "1coi_C.pdb")

@fixture
def c1pose(pdbdir):
   return get_pose(pdbdir, "c1.pdb")

@fixture
def c2pose(pdbdir):
   return get_pose(pdbdir, "c2.pdb")

@fixture
def c3pose(pdbdir):
   return get_pose(pdbdir, "c3.pdb")

@fixture
def c3hetpose(pdbdir):
   return get_pose(pdbdir, "c3het.pdb")

@fixture
def c3_splay_pose(pdbdir):
   return get_pose(pdbdir, "c3_splay.pdb")

@fixture
def c4pose(pdbdir):
   return get_pose(pdbdir, "c4.pdb")

@fixture
def c5pose(pdbdir):
   return get_pose(pdbdir, "c5.pdb")

@fixture
def c6pose(pdbdir):
   return get_pose(pdbdir, "c6.pdb")

@fixture
def db_asu_pose(pdbdir):
   return get_pose(pdbdir, "db_asu.pdb")

@fixture
def hetC2A_pose(pdbdir):
   return get_pose(pdbdir, "hetC2A.pdb")

@fixture
def hetC2B_pose(pdbdir):
   return get_pose(pdbdir, "hetC2B.pdb")
