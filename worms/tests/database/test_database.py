from tempfile import _TemporaryFileWrapper
import pytest
from worms.database import *
import logging
import json
from worms.util import InProcessExecutor
from pprint import pprint
from os.path import dirname
import numba as nb
from worms.tests import only_if_pyrosetta, only_if_pyrosetta_distributed

@only_if_pyrosetta_distributed
def test_construct_database_from_pdbs(tmpdir, jsondir):
   pp = CachingBBlockDB(
      cachedirs=str(tmpdir),
      dbfiles=[os.path.join(jsondir, "test_db_file.json")],
      lazy=False,
      read_new_pdbs=True,
   )

   # print([len(pp.pose(k)) for k in pp.query_names('all')])
   assert [len(pp.pose(k)) for k in pp.query_names("all")] == [
      13, 24, 27, 27, 27, 40, 35, 36, 13, 8, 7, 9
   ]
   keys = sorted(pp._bblock_cache.keys(), key=lambda x: pp._key_to_pdbfile[x])
   assert np.all(pp.bblock(keys[0]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[0]).conn_resids(1) == [12])
   assert np.all(pp.bblock(keys[1]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[1]).conn_resids(1) == [11])
   assert np.all(pp.bblock(keys[2]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[2]).conn_resids(1) == [8])
   assert np.all(pp.bblock(keys[3]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[3]).conn_resids(1) == [6, 7, 8])
   assert np.all(pp.bblock(keys[3]).conn_resids(2) == [9])
   assert np.all(pp.bblock(keys[3]).conn_resids(3) == [16, 17])
   assert np.all(pp.bblock(keys[3]).conn_resids(4) == [18, 19, 20])
   assert np.all(pp.bblock(keys[3]).conn_resids(5) == [26])
   assert np.all(pp.bblock(keys[4]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[4]).conn_resids(1) == [8])
   assert np.all(pp.bblock(keys[4]).conn_resids(2) == [9])
   assert np.all(pp.bblock(keys[4]).conn_resids(3) == [17])
   assert np.all(pp.bblock(keys[4]).conn_resids(4) == [18])
   assert np.all(pp.bblock(keys[4]).conn_resids(5) == [26])
   assert np.all(pp.bblock(keys[5]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[5]).conn_resids(1) == [9])
   assert np.all(pp.bblock(keys[6]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[6]).conn_resids(1) == [6])
   assert np.all(pp.bblock(keys[7]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[7]).conn_resids(1) == [5])
   assert np.all(pp.bblock(keys[8]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[8]).conn_resids(1) == [6, 7, 8, 9, 10, 11, 12])
   assert np.all(pp.bblock(keys[9]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[9]).conn_resids(1) == [7])
   assert np.all(pp.bblock(keys[10]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[10]).conn_resids(1) == [6])
   assert np.all(pp.bblock(keys[11]).conn_resids(0) == [0])
   assert np.all(pp.bblock(keys[11]).conn_resids(1) == [8])

def test_conftest_pdbfile(db_bblock_caching_v0):
   assert len(db_bblock_caching_v0.query("all")) == 12

if __name__ == '__main__':
   from tempfile import TemporaryDirectory
   tmp = TemporaryDirectory()
   tmpdir = tmp.name

   test_construct_database_from_pdbs(tmpdir, worms.data.json_dir)
   # test_splicedb_merge()
   # from worms.tests.conftest import db_bblock_caching_v0
   # test_conftest_pdbfile(db_bblock_caching_v0())