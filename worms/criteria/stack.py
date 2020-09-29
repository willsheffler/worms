from .base import WormCriteria

import numpy as np

from worms.util import jit

class Stack(WormCriteria):
   """
    """
   def __init__(self, sym, *, from_seg=0, tolerance=1.0, lever=50, to_seg=-1):
      if from_seg == to_seg:
         raise ValueError("from_seg should not be same as to_seg")
      self.sym = sym
      self.from_seg = from_seg
      self.tolerance = tolerance
      self.lever = lever
      self.to_seg = to_seg
      self.rtol = tolerance / lever
      self.is_cyclic = False
      self.symname = "C" + str(self.sym)
      self.origin_seg = None

   def score(self):
      raise NotImplementedError

   def jit_lossfunc(self):
      from_seg = self.from_seg
      to_seg = self.to_seg
      ctol_sq = self.tolerance**2
      rtol = self.rtol

      @jit
      def func(pos, idx, verts):
         cen2 = pos[to_seg, :, 3]
         ax2 = pos[to_seg, :, 2]
         dist_sq = cen2[0]**2 + cen2[1]**2
         angl = np.arccos(np.abs(ax2[2]))
         err_sq = (angl / rtol)**2
         err_sq += dist_sq / ctol_sq
         return np.sqrt(err_sq)

      return func

   def alignment(self, segpos, debug=0, **kw):
      return np.eye(4)

   def merge_segment(self, **kw):
      return None

   def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
      "return spearate criteria for each search stage"
      return [(self, bbs)], None

   # def cloned_segments(self):
   # "which bbs are being merged together"
   # return (self.from_seg, )

   def iface_rms(self, pose0, prov, **kw):
      return -1
