import numpy as np
from worms import util
from worms.bunch import Bunch
from homog import numba_axis_angle, hrot, angle

def foo():
   pass

def make_const_jitfunc(v):
   @util.jit
   def dummy(*_):
      return v

   return dummy

def make_helixconf_filter(**kw):
   kw = Bunch(**kw)

   if 0 == kw.helixconf_min_num_horiz_helix:
      return make_const_jitfunc(0)

   zpad = kw.helixconf_max_depth_within_hull
   min_perp_helix = kw.helixconf_min_num_horiz_helix
   helix_max_cos = np.cos(np.radians(90 - kw.helixconf_max_vert_angle))
   print('helix_max_cos', helix_max_cos)

   @util.jit
   def func(pos, idx, verts, xhat, axis, angle):
      zlb, zub = 9e9, -9e9
      for iv in range(len(verts) - 1):
         v = verts[iv]
         xseg = pos[iv]
         vidx = idx[iv]
         ibb = v.ibblock[vidx]
         for ihull in range(v.numhull[ibb]):
            pt = xseg @ v.hull[ibb, ihull]
            z = np.sum(axis * pt)
            zlb = min(z, zlb)
            zub = max(z, zub)
      zlb += zpad
      zub -= zpad
      # print('ZBOUNDS', zlb, zub)

      hsupper, hslower = 0, 0
      for iv in range(len(verts) - 1):
         v = verts[iv]
         xseg = pos[iv]
         vidx = idx[iv]
         ibb = v.ibblock[vidx]
         # xbb =
         # print(iv, vidx, ibb, v.numhelix[ibb])
         for ih in range(v.numhelix[ibb]):
            beg = xseg @ v.helixbeg[ibb, ih, :]
            end = xseg @ v.helixend[ibb, ih, :]
            z = np.sum(axis * (beg + end) / 2)
            hvec = end - beg
            dot = abs(np.sum(axis * hvec) / np.sqrt(np.sum(hvec * hvec)))
            # print('foo', iv, ih, dot, dot <= helix_max_cos)
            if dot > helix_max_cos:
               continue
            # print('zbound', zlb, z, zub)
            if z < zlb: hslower += 1
            if z > zub: hsupper += 1
            # print('ZVAL', z)
            # print(iv, vidx, ibb, ih, beg, end)

      # helix_score = max(hsupper, hslower)
      helix_score = max(hsupper, hslower)

      # print('helix_score', helix_score)
      # assert 0
      if helix_score < min_perp_helix:
         return 9e9
      return 0.0

   return func
