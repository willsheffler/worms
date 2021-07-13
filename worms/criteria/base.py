import abc
import numpy as np
import homog as hm
from numpy.linalg import inv
from worms.util import jit

Ux = np.array([1, 0, 0, 0])
Uy = np.array([0, 1, 0, 0])
Uz = np.array([0, 0, 1, 0])

class WormCriteria(abc.ABC):
   @abc.abstractmethod
   def score(self, **kw):
      pass

   allowed_attributes = (
      "last_body_same_as",
      "symname",
      "is_cyclic",
      "alignment",
      "from_seg",
      "to_seg",
      "origin_seg",
      "symfile_modifiers",
      "crystinfo",
   )

class CriteriaList(WormCriteria):
   def __init__(self, children):
      if isinstance(children, WormCriteria):
         children = [children]
      self.children = children

   def score(self, **kw):
      return sum(c.score(**kw) for c in self.children)

   def __getattr__(self, name):
      if name not in WormCriteria.allowed_attributes:
         raise AttributeError("CriteriaList has no attribute: " + name)
      r = [getattr(c, name) for c in self.children if hasattr(c, name)]
      r = [x for x in r if x is not None]
      assert len(r) < 2
      return r[0] if len(r) else None

   def __getitem__(self, index):
      assert isinstance(index, int)
      return self.children[index]

   def __len__(self):
      return len(self.children)

   def __iter__(self):
      return iter(self.children)

from worms.filters.helixconf_jit import make_helixconf_filter

class NullCriteria(WormCriteria):
   def __init__(self, from_seg=0, to_seg=-1, origin_seg=None):
      self.from_seg = from_seg
      self.to_seg = to_seg
      self.origin_seg = None
      self.is_cyclic = False
      self.tolerance = 9e8
      self.symname = None

   def merge_segment(self, **kw):
      return None

   def score(self, segpos, **kw):
      return np.zeros(segpos[-1].shape[:-2])

   def alignment(self, segpos, **kw):
      r = np.empty_like(segpos[-1])
      r[..., :, :] = np.eye(4)
      return r

   def jit_lossfunc(self, **kw):
      kw = Bunch(**kw)

      from_seg = self.from_seg
      to_seg = self.to_seg
      lever = self.lever
      min_sep2 = self.min_sep2

      helixconf_filter = make_helixconf_filter(**kw)

      @jit
      def null_lossfunc(pos, idx, verts):

         x_from = pos[from_seg]
         x_to = pos[to_seg]
         xhat = x_to @ np.linalg.inv(x_from)
         if np.sum(xhat[:3, 3]**2) < min_sep2:
            return 9e9
         axis = np.array([0, 0, 1, 0])

         helixerr = helixconf_filter(pos, idx, verts, xhat, axis)
         return helixerr

      return null_lossfunc

   def iface_rms(self, pose0, prov0, **kw):
      return -1
