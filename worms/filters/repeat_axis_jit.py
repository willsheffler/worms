import numpy as np
from worms import util
from worms.util import Bunch
from worms.homog import numba_axis_angle_cen, numba_cross
from worms.util.jitutil import make_const_jitfunc

def make_repeat_axis_filter(criteria, **kw):
   kw = Bunch(kw)
   if kw.repeat_axis_check < 0:
      return make_const_jitfunc(0)

   repeat_axis_check = kw.repeat_axis_check
   repeat_axis_weight = kw.repeat_axis_weight

   @util.jit
   def func(pos, idx, verts, axis1, axis2):

      iv = repeat_axis_check
      if iv < 0: return 0
      vidx = idx[iv]
      pos = pos[iv]
      ibb = verts[iv].ibblock[vidx]
      repeataxis = verts[iv].repeataxis[ibb]
      repeataxis = pos @ repeataxis
      assert repeataxis[3] == 0
      repeataxis = repeataxis / np.sqrt(np.sum(repeataxis**2))
      perp = numba_cross(axis1, axis2)
      return repeat_axis_weight * np.abs(np.sum(repeataxis * perp))
      # return 10

   return func

def make_const_jitfunc(v):
   @util.jit
   def dummy(*_):
      return v

   return dummy
