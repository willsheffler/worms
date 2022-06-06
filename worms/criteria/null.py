from worms.criteria import *
from worms import util
from worms.homog import numba_axis_angle, hrot
# from xbin import gu_xbin_indexer, numba_xbin_indexer
from copy import deepcopy
from worms.util import jit

class NullCriteria(WormCriteria):
   def __init__(self, cyclic=1, **kw):
      super().__init__(**kw)
      self.from_seg = 0
      self.to_seg = -1
      self.origin_seg = None
      self.is_cyclic = False
      self.tolerance = 9e8
      self.symname = 'C%i' % cyclic if cyclic > 1 else None

   def merge_segment(self, **kw):
      return None

   def score(self, segpos, **kw):
      return np.zeros(segpos[-1].shape[:-2])

   def alignment(self, segpos, **kw):
      r = np.empty_like(segpos[-1])
      r[..., :, :] = np.eye(4)
      return r

   def jit_lossfunc(self, **kw):
      helixconf_filter = worms.filters.helixconf_jit.make_helixconf_filter(self, **kw)

      @jit
      def null_lossfunc(pos, idx, verts):
         axis = np.array([0, 0, 1, 0])
         cen = np.array([0, 0, 0, 1])
         helixerr = helixconf_filter(pos, idx, verts, axis)
         # if helixerr < 9e8:
         # print('null_lossfunc', helixerr)
         return helixerr
         # return 0.0

      return null_lossfunc

   def iface_rms(self, pose0, prov0, **kw):
      return -1

   def __eq__(self, other):
      return all([
         self.from_seg == other.from_seg,
         self.to_seg == other.to_seg,
         self.origin_seg == other.origin_seg,
         self.is_cyclic == other.is_cyclic,
         self.tolerance == other.tolerance,
         self.symname == other.symname,
      ])

# class Null(WormCriteria):
#    def __init__(
#       self,
#       from_seg=0,
#       origin_seg=None,
#       to_seg=-1,
#       **kw,
#    ):
#       super().__init__(**kw)
#       self.from_seg = from_seg
#       self.to_seg = to_seg
#       self.origin_seg = origin_seg

#    def score(self, segpos, *, verbosity=False, **kw):
#       return 0.0

#    def alignment(self, segpos, **kw):
#       return np.eye(4)

#    def jit_lossfunc(self, **kw):
#       @util.jit
#       def func(pos, idx, verts):
#          return 0.0

#       return func

#    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
#       return [(self, bbs)], None

#    def merge_segment(self, **kw):
#       return self.from_seg

#    def cloned_segments(self):
#       "which bbs are being merged together"
#       return self.from_seg, self.to_seg

#    def iface_rms(self, pose0, prov, **kw):
#       return -1
