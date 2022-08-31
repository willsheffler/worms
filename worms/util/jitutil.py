import numba, numpy as np
from worms.util.util import jit
from worms.search.result import ResultJIT

def make_const_jitfunc(v):
   @jit
   def dummy(*_):
      return v

   return dummy

@jit
def binary_search_pair(is_sorted, tgt, ret=0):
   n = len(is_sorted)
   if n == 1:
      if is_sorted[0, 0] == tgt[0] and is_sorted[0, 1] == tgt[1]:
         return ret
      else:
         return -1
   mid = n // 2
   if (is_sorted[mid, 0], is_sorted[mid, 1]) > tgt:
      return binary_search_pair(is_sorted[:mid], tgt, ret)
   else:
      return binary_search_pair(is_sorted[mid:], tgt, ret + mid)

@jit
def expand_array_if_needed(ary, i):
   if len(ary) > i:
      return ary
   newshape = (ary.shape[0] * 2, ) + ary.shape[1:]
   new = np.zeros(newshape, dtype=ary.dtype) - ary.dtype.type(1)
   new[:len(ary)] = ary
   return new

@jit
def expand_results(result, nresults):
   if len(result[0]) <= nresults:
      result = ResultJIT(
         expand_array_if_needed(result[0], nresults),
         expand_array_if_needed(result[1], nresults),
         expand_array_if_needed(result[2], nresults),
         result.stats,
      )
   result.idx[nresults] = result.idx[nresults - 1]
   result.pos[nresults] = result.pos[nresults - 1]
   result.err[nresults] = result.err[nresults - 1]
   return result

@numba.njit("int32[:](int32[:])", nogil=1)
def contig_idx_breaks(idx):
   breaks = np.empty(idx[-1] + 2, dtype=np.int32)
   breaks[0] = 0
   n = 1
   for i in range(1, len(idx)):
      if idx[i - 1] != idx[i]:
         assert idx[i - 1] < idx[i]
         breaks[n] = i
         n += 1
   breaks[n] = len(idx)
   breaks = np.ascontiguousarray(breaks[:n + 1])
   if __debug__:
      for i in range(breaks.size - 1):
         vals = idx[breaks[i]:breaks[i + 1]]
         assert len(vals)
         assert np.all(vals == vals[0])
   return breaks

@jit
def _unique_key_int32s(keys):
   map = -np.ones(np.max(keys) + 1, dtype=np.int32)
   count = 0
   for k in keys:
      if map[k] < 0:
         map[k] = count
         count += 1
   out = np.empty(len(keys), dtype=np.int32)
   for i in range(len(keys)):
      out[i] = map[keys[i]]
   return out

def unique_key_int32s(a, b):
   if b[0] == -1:
      assert np.all(b == -1)
      return a
   a = a.astype("i8")
   b = b.astype("i8")
   m = np.max(a) + 1
   k = b * m + a
   assert np.all(k >= 0)
   return _unique_key_int32s(k)

@numba.njit("int32[:](int32[:])", nogil=1)
def contig_idx_breaks(idx):
   breaks = np.empty(idx[-1] + 2, dtype=np.int32)
   breaks[0] = 0
   n = 1
   for i in range(1, len(idx)):
      if idx[i - 1] != idx[i]:
         assert idx[i - 1] < idx[i]
         breaks[n] = i
         n += 1
   breaks[n] = len(idx)
   breaks = np.ascontiguousarray(breaks[:n + 1])
   if __debug__:
      for i in range(breaks.size - 1):
         vals = idx[breaks[i]:breaks[i + 1]]
         assert len(vals)
         assert np.all(vals == vals[0])
   return breaks
