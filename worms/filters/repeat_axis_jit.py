import numpy as np
from worms import util
from worms.util import Bunch
from worms.homog import numba_axis_angle_cen, hrot, angle
from worms.util.jitutil import make_const_jitfunc

def make_repeat_axis_filter(criteria, **kw):
   kw = Bunch(kw)
   if kw.repeat_axis_check < 0:
      return make_const_jitfunc(0)

   repeat_axis_check = kw.repeat_axis_check

   @util.jit
   def func(pos, idx, verts, axis1, axis2=None):
      verts.bbs
      return 13.0

   return func

def make_const_jitfunc(v):
   @util.jit
   def dummy(*_):
      return v

   return dummy
