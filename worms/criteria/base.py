import abc
import numpy as np
from worms import homog as hm
from numpy.linalg import inv
from worms.util import jit
import worms
# from worms.filters.helixconf_jit import make_helixconf_filter

Ux = np.array([1, 0, 0, 0])
Uy = np.array([0, 1, 0, 0])
Uz = np.array([0, 0, 1, 0])

class WormCriteria(abc.ABC):
   def __init__(self, bbspec=None):
      self.bbspec = bbspec

   @abc.abstractmethod
   def score(self, **kw):
      pass

   def symops(self, segpos):
      return None

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
   def __init__(
      self,
      children,
      **kw,
   ):
      super().__init__(**kw)
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
