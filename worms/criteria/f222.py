from . import WormCriteria
import numpy as np
from worms import homog as hm
from worms.util import jit

class F222_D2_D2(WormCriteria):
   def __init__(
      self,
      d2a=0,
      d2b=-1,
      min_cell_size=20,
      max_cell_size=9999,
      **kw,
   ):
      super().__init__(**kw)

      assert d2a is 0
      self.from_seg = d2a
      self.to_seg = d2b
      self.space_group_str = "F 2 2 2"
      self.symname = "F222_D2_D2_3"
      self.tolerance = 1.0  # will get set elsewhere
      self.lever = 50  # will get set elsewhere
      self.is_cyclic = False
      self.origin_seg = None
      self.min_cell_size = min_cell_size
      self.max_cell_size = max_cell_size

   def score(self, segpos, **kw):
      raise NotImplementedError

   def jit_lossfunc(self, **kw):
      to_seg = self.to_seg
      tolerance = self.tolerance
      lever = self.lever
      origin = np.array([0, 0, 0])
      minsep = self.min_cell_size
      maxsep = self.max_cell_size

      @jit
      def lossfunc(pos, idx, vrts):
         "pos xyz axes should align along permutation of x y z"
         ax = pos[to_seg, :3, 0]
         ay = pos[to_seg, :3, 1]
         cn = pos[to_seg, :3, 3]
         axi = np.argmax(np.abs(ax))
         ayi = np.argmax(np.abs(ay))
         if axi == ayi or np.any(np.abs(cn) < minsep) or np.any(np.abs(cn) > maxsep):
            return 9e9
         angerr = np.arccos(min(abs(ax[axi]), abs(ay[ayi])))
         return angerr * lever

      return lossfunc

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      ax = segpos[self.to_seg, :3, 0]
      ay = segpos[self.to_seg, :3, 1]
      cn = segpos[self.to_seg, :3, 3]

      # print("alignment")
      # print("   ", cn, ax, ay)

      xalign = np.eye(4)
      cell_spacing = np.abs(cn)

      # permute and update cell_spacing ??
      # only worth it with smarter symdef file

      if out_cell_spacing:
         return xalign, cell_spacing
      else:
         return xalign

   def symfile_modifiers(self, segpos):
      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      return dict(scale_positions=cell_dist)

   def crystinfo(self, segpos):
      # CRYST1   85.001   85.001   85.001  90.00  90.00  90.00 P 21 3
      if self.space_group_str is None:
         return None
      # print("hi")
      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      return (*cell_dist * 4, 90, 90, 90, self.space_group_str)

   def merge_segment(self, **kw):
      return self.from_seg

   def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
      "return spearate criteria for each search stage"
      return [(self, bbs)], None

   def cloned_segments(self):
      "which bbs are being merged together"
      return (self.from_seg, )

   def iface_rms(self, pose0, prov, **kw):
      return -1
