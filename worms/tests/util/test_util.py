import json, itertools, pytest, numpy as np
from worms.util import jitutil as util

def test_unique_key_int32s():
   a = np.array([1, 4, 7], dtype=np.int32)
   k = util._unique_key_int32s(a)
   assert list(k) == [0, 1, 2]

@pytest.mark.skip()
def test_infer_symmetry(c1pose, c2pose, c3pose, c3hetpose, c6pose):
   print(c3pose)
   assert 0

# def test_MultiRange():
# mr = util.MultiRange([2, 3, 4, 2, 3])
# prod = itertools.product(*[range(n) for n in mr.nside])
# for i, tup in enumerate(prod):
# assert tup == mr[i]
# assert i + 1 == len(mr)

@pytest.mark.skip()
def test_remove_dicts():
   jd = json.load(open(test_db_files[0]))[0]
   ji = dicts_to_items(jd)
   assert jd == items_to_dicts(ji)
   assert ji == dicts_to_items(jd)
   assert isinstance(jd, dict)
   assert isinstance(ji, list)
   assert isinstance(ji[0], tuple)
   assert len(ji[0]) == 2

def test_contig_idx_breaks():
   tst = np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype="i4")
   assert np.all(util.contig_idx_breaks(tst) == [0, 4, 8])

def test_numba_expand_array_if_needed_1d():
   ary0 = ary = np.arange(7)
   for i in range(7):
      ary = util.expand_array_if_needed(ary, i)
   assert ary0 is ary
   for i in range(7, 100):
      ary = util.expand_array_if_needed(ary, i)
   assert len(ary.shape) == 1
   assert ary.shape[0] >= 100
   assert np.all(ary[:len(ary0)] == ary0)
   assert np.all(ary[len(ary0):] == -1)

def test_numba_expand_array_if_needed_2d1():
   ary0 = ary = np.arange(7).reshape((7, 1))
   for i in range(7):
      ary = util.expand_array_if_needed(ary, i)
   assert ary0 is ary
   for i in range(7, 100):
      ary = util.expand_array_if_needed(ary, i)
   assert len(ary.shape) == 2
   assert ary.shape[0] >= 100
   assert ary.shape[1] == 1
   assert np.all(ary[:len(ary0)] == ary0)
   assert np.all(ary[len(ary0):] == -1)

def test_numba_expand_array_if_needed_2d():
   ary0 = ary = np.stack([np.arange(7)] * 2, axis=1)
   for i in range(7):
      ary = util.expand_array_if_needed(ary, i)
   assert ary0 is ary
   for i in range(7, 100):
      ary = util.expand_array_if_needed(ary, i)
   assert len(ary.shape) == 2
   assert ary.shape[0] >= 100
   assert ary.shape[1] == 2
   assert np.all(ary[:len(ary0)] == ary0)
   assert np.all(ary[len(ary0):] == -1)

def test_numba_expand_array_if_needed_7d():
   ary0 = ary = np.stack([np.arange(7)] * 7, axis=1)
   for i in range(7):
      ary = util.expand_array_if_needed(ary, i)
   assert ary0 is ary
   for i in range(7, 100):
      ary = util.expand_array_if_needed(ary, i)
   assert len(ary.shape) == 2
   assert ary.shape[0] >= 100
   assert ary.shape[1] == 7
   assert np.all(ary[:len(ary0)] == ary0)
   assert np.all(ary[len(ary0):] == -1)

def test_numba_expand_array_if_needed_int32():
   ary0 = ary = np.arange(7).astype(np.int32)
   for i in range(7):
      ary = util.expand_array_if_needed(ary, i)
   assert ary0 is ary
   for i in range(7, 100):
      ary = util.expand_array_if_needed(ary, i)
   assert len(ary.shape) == 1
   assert ary.shape[0] >= 100
   assert np.all(ary[:len(ary0)] == ary0)
   assert np.all(ary[len(ary0):] == -1)

if __name__ == '__main__':
   test_unique_key_int32s()
   # test_remove_dicts()
   test_contig_idx_breaks()
   test_numba_expand_array_if_needed_1d()
   test_numba_expand_array_if_needed_2d1()
   test_numba_expand_array_if_needed_2d()
   test_numba_expand_array_if_needed_7d()
   test_numba_expand_array_if_needed_int32()
