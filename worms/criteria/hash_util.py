# raise NotImplementedError('khash_cffi needs updating for numba 0.49+')

import numpy as np

from worms import homog as hg

from worms.criteria import WormCriteria
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.util import jit

@jit
def encode_indices(sizes, indices):
   prod = np.int64(1)
   index = np.int64(0)
   for i in range(len(sizes)):
      index += prod * indices[i]
      prod *= sizes[i]
   return index

@jit
def decode_indices(sizes, index):
   indices = np.zeros(sizes.shape, dtype=np.int64)
   for i, size in enumerate(sizes):
      indices[i] = index % size
      index -= index % size
      index /= size
   assert index == 0
   return indices

def make_hash_table(ssdag, rslt, gubinner):
   if len(rslt.idx) is 0:
      return None, None
   assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
   assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
   assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
   keys = gubinner(rslt.pos[:, -1])
   assert keys.dtype == np.int64
   ridx = np.arange(len(rslt.idx))
   ibb0 = ssdag.verts[+0].ibblock[rslt.idx[:, +0]]
   ibb1 = ssdag.verts[-1].ibblock[rslt.idx[:, -1]]
   isite0 = ssdag.verts[+0].isite[rslt.idx[:, +0], 1]
   isite1 = ssdag.verts[-1].isite[rslt.idx[:, -1], 0]
   assert np.all(ibb0 == ibb1)
   assert np.all(isite0 != isite1)
   assert np.all(isite0 < 2**8)
   assert np.all(isite1 < 2**8)
   assert np.all(ibb0 < 2**16)
   assert np.all(keys >= 0)
   vals = (np.left_shift(ridx, 32) + np.left_shift(ibb0, 16) + np.left_shift(isite0, 8) + isite1)
   # print(keys[:10])
   # print(vals[:10])
   hash_table = KHashi8i8()
   hash_table.update2(keys, vals)
   return keys, hash_table

class WheelHashCriteria(WormCriteria):
   def __init__(
      self,
      orig_criteria,
      binner,
      hash_table,
      **kw,
   ):
      super().__init__(**kw)
      self.orig_criteria = orig_criteria
      self.binner = binner
      self.hash_table = hash_table
      self.is_cyclic = False

   def cloned_segments(self):
      return (-1, )

   def score(self, *args):
      return 0

   def jit_lossfunc(self, **kw):
      rots, irots = _get_hash_lossfunc_data(self.orig_criteria.nfold)
      binner = self.binner
      hash_vp = self.hash_table.hash

      @jit
      def func(pos, idx, verts):
         pos = pos[-1]
         for irot in irots:
            to_pos = rots[irot] @ pos
            xtgt = (np.linalg.inv(pos) @ to_pos).astype(np.float64)
            key = binner(xtgt)
            val = _khash_get(hash_vp, key, -123456789)

            # if missing, no hit
            if val == -123456789:
               continue

            # must use same bblock
            ibody = verts[-1].ibblock[idx[-1]]
            ibody0 = np.right_shift(val, 16) % 2**16
            if ibody != ibody0:
               continue

            # must use different site
            isite0 = np.right_shift(val, 8) % 2**8
            isite1 = val % 2**8
            isite = verts[-1].isite[idx[-1], 0]
            if isite == isite0 or isite == isite1:
               continue

            return 0
         return 9e9

      return func

def _get_hash_lossfunc_data(nfold):
   rots = np.stack((
      hg.hrot([0, 0, 1, 0], np.pi * 2.0 / nfold),
      hg.hrot([0, 0, 1, 0], -np.pi * 2.0 / nfold),
   )).astype(np.float32)
   assert rots.shape == (2, 4, 4)
   irots = (0, 1) if nfold > 2 else (0, )
   return rots, irots

def _get_hash_val(gubinner, hash_table, pos, nfold):
   rots, irots = _get_hash_lossfunc_data(nfold)
   for irot in irots:
      to_pos = rots[irot] @ pos
      xtgt = np.linalg.inv(pos) @ to_pos
      xtgt = xtgt.astype(np.float64)
      key = gubinner(xtgt)
      val = hash_table.get(key)
      if val != -123456789:
         return val
   # assert 0, 'pos not found in table!'
   return -1
