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
   print('%' * 80)
   if 'NUMBA_DISABLE_JIT' in os.environ:
      print(os.environ['NUMBA_DISABLE_JIT'])
   else:
      print('NUMBA_DISABLE_JIT not in os.environ')
   print('%' * 80, flush=True)
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
   assert np.all(pp.bblock(keys[0]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[0]).conn_resids(1) == [12])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[1]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[1]).conn_resids(1) == [11])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[2]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[2]).conn_resids(1) == [8])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(1) == [6, 7, 8])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(2) == [9])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(3) == [16, 17])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(4) == [18, 19, 20])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[3]).conn_resids(5) == [26])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(1) == [8])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(2) == [9])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(3) == [17])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(4) == [18])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[4]).conn_resids(5) == [26])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[5]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[5]).conn_resids(1) == [9])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[6]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[6]).conn_resids(1) == [6])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[7]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[7]).conn_resids(1) == [5])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[8]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[8]).conn_resids(1) == [6, 7, 8, 9, 10, 11,
                                                       12])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[9]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[9]).conn_resids(1) == [7])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[10]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[10]).conn_resids(1) == [6])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[11]).conn_resids(0) == [0])  #  # pyright: ignore
   assert np.all(pp.bblock(keys[11]).conn_resids(1) == [8])  #  # pyright: ignore

def test_conftest_pdbfile(db_bblock_caching_v0):
   assert len(db_bblock_caching_v0.query("all")) == 12

if __name__ == '__main__':
   from tempfile import TemporaryDirectory
   with TemporaryDirectory() as tmpdir:

      test_construct_database_from_pdbs(tmpdir, worms.data.json_dir)
   # test_splicedb_merge()
   # from worms.tests.conftest import db_bblock_caching_v0
   # test_conftest_pdbfile(db_bblock_caching_v0())
   print('$$$$$$$$$$$$ done $$$$$$$$$$$$$$$')
