import numpy as np
import worms
import willutil as wu
from worms.homog import numba_axis_angle_cen, numba_cross
from worms.util.jitutil import make_const_jitfunc

def make_repeat_axis_filter_axesisect(criteria, **kw):
   kw = wu.Bunch(kw)
   if kw.repeat_axis_check < 0:
      return make_const_jitfunc(0)

   repeat_axis_check = kw.repeat_axis_check
   repeat_axis_weight = kw.repeat_axis_weight

   assert isinstance(criteria, worms.criteria.AxesIntersect)

   @worms.util.jit
   def func(pos, idx, verts, axis1, axis2):
      iv = repeat_axis_check
      if iv < 0: return 0
      vidx = idx[iv]
      pos = pos[iv]
      ibb = verts[iv].ibblock[vidx]
      repeataxis = verts[iv].repeataxis[ibb]
      repeataxis = pos @ repeataxis
      assert repeataxis[3] == 0  # sanity check is vector
      repeataxis = repeataxis / np.sqrt(np.sum(repeataxis**2))
      perp = numba_cross(axis1, axis2)
      return repeat_axis_weight * np.abs(np.sum(repeataxis * perp))
      # return 10

   return func

def make_repeat_axis_filter_cyclic(criteria, **kw):
   kw = wu.Bunch(kw)
   if kw.repeat_axis_check < 0:
      return make_const_jitfunc(0)
   assert isinstance(criteria, worms.criteria.Cyclic)
   repeat_axis_check = kw.repeat_axis_check
   repeat_axis_weight = kw.repeat_axis_weight

   @worms.util.jit
   def func(pos, idx, verts, cyclic_axis):
      iv = repeat_axis_check
      if iv < 0: return 0
      vidx = idx[iv]
      pos = pos[iv]
      ibb = verts[iv].ibblock[vidx]
      repeataxis = verts[iv].repeataxis[ibb]
      repeataxis = pos @ repeataxis
      assert repeataxis[3] == 0  # sanity check is vector
      repeataxis = repeataxis / np.sqrt(np.sum(repeataxis**2))
      perp = abs(np.sum(repeataxis * cyclic_axis))
      return repeat_axis_weight * perp

   return func

def make_const_jitfunc(v):
   @util.jit
   def dummy(*_):
      return v

   return dummy
