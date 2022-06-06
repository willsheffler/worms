from .base import WormCriteria

import numpy as np

from worms.util import jit

class Stack(WormCriteria):
   """
    """
   def __init__(self, sym, *, from_seg=0, tolerance=1.0, lever=50, to_seg=-1, **kw):
      super().__init__(**kw)
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

   def jit_lossfunc(self, **kw):
      from_seg = self.from_seg
      to_seg = self.to_seg
      ctol_sq = self.tolerance**2
      rtol = self.rtol

      @jit
      def func(pos, idx, verts):
         cen2 = pos[to_seg, :, 3].copy()  #  this was a good bug!
         ax2 = pos[to_seg, :, 2]
         cen2[2] = 0.0
         dist2 = np.sum(cen2**2)
         ang2 = np.arccos(np.abs(ax2[2]))**2
         err = np.sqrt(ang2 / rot_tol2 + dist2 / tol2)
         return err

      # try to correlate angle/cart deviation

      # cen = pos[to_seg, :3, 3]
      # # cen = cen / np.linalg.norm(cen)
      # axis = pos[to_seg, :3, 2]
      # if np.sum(axis * cen) < 0:
      #    axis = -axis

      # dist_sq = cen[0]**2 + cen[1]**2
      # if dist_sq > ctol_sq * 4: return 9e9

      # axis_angle = np.arccos(np.abs(axis[2]))
      # if axis_angle > rtol * 4: return 9e9

      # # cart_angle = np.arccos(np.abs(cen[2]/np.linalg.norm(cen)))
      # # cart_perp = np.array([cen[0], cen[1], 0])
      # correction_axis = np.array([axis[1], -axis[0], 0])  # cross prod w/z

      # correction_axis = correction_axis / np.linalg.norm(correction_axis)
      # cart_bad_err = np.abs(np.sum(correction_axis * cen))

      # cen_len = np.linalg.norm(cen)
      # axis_to_cart = axis * cen_len
      # delta = axis_to_cart - cen

      # return np.sqrt(np.sum(delta**2) / ctol_sq)  #+ cart_bad_err

      # ang_err2 = (axis_angle / rtol)**2
      # dist_sq = cen[0]**2 + cen[1]**2
      # dis_errg2= dist_sq / ctol_sq
      # return np.sqrt(err_sq)

      # cen2 = pos[to_seg, :, 3]
      #  ax2 = pos[to_seg, :, 2]
      #  dist = np.sqrt(cen2[0]**2 + cen2[1]**2)
      #  ang = np.arccos(np.abs(ax2[2]))
      #  err = ang / rot_tol + dist / tol
      #  return err

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
