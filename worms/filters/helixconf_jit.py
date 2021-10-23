import numpy as np, random
from worms import util
from worms.util import Bunch
from worms.homog import numba_axis_angle_cen, hrot, angle

def make_helixconf_filter(criteria, **kw):
   kw = Bunch(**kw)

   if 0 == kw.helixconf_min_num_horiz_helix:
      return make_const_jitfunc(0)

   zpad = kw.helixconf_max_depth_within_hull
   min_perp_helix = kw.helixconf_min_num_horiz_helix
   helix_max_cos = np.cos(np.radians(90 - kw.helixconf_max_vert_angle))
   print('helix_max_cos', helix_max_cos)
   trim_bblocks = 1 if criteria.is_cyclic else 0
   use_hull = kw.helixconf_use_hull_for_surface

   min_axis_dist = kw.helixconf_min_dist_to_sym_axis
   max_axis_dist = kw.helixconf_max_dist_to_sym_axis

   @util.jit
   def func(pos, idx, verts, axis):
      # def func(pos, idx, verts, axis, cen):
      # if random.random() < .9: return 9e9

      zlb, zub = _get_z_bounds(pos, idx, verts, axis, trim_bblocks, use_hull, zpad)

      hsupper, hslower = 0, 0
      for iv in range(len(verts) - trim_bblocks):
         v = verts[iv]
         xseg = pos[iv]
         vidx = idx[iv]
         ibb = v.ibblock[vidx]
         reslb, resub = _get_resbounds(v, vidx)

         # print('    INFO', iv, vidx, ibb, v.numhelix[ibb], reslb, resub)
         # assert 0
         for ih in range(v.numhelix[ibb]):
            beg, end, zbeg, zend = _get_coords_resbound(v, ibb, ih, reslb, resub, xseg, axis)
            if zbeg == -12345.0: continue
            hvec = end - beg
            hvec = hvec / np.sqrt(np.sum(hvec**2))
            dot = abs(np.sum(axis * hvec))
            # print('        hrelixres', begres, endres, dot)
            # print('foo', iv, ih, dot, dot <= helix_max_cos, hvec)
            if dot > helix_max_cos:
               continue
            # print('zbound', zlb, z, zub)
            if zbeg < zlb or zend < zlb: hslower += 1
            if zbeg > zub or zend > zub: hsupper += 1
            # print('ZVAL', z)
            # print(iv, vidx, ibb, ih, beg, end)

      # helix_score = max(hsupper, hslower)
      helix_score = max(hsupper, hslower)

      # if helix_score > 0:
      # print('    ZBOUNDS', zlb, zub, 'HELIX', hsupper, hslower)
      # assert 0

      # print('helix_score', helix_score)
      # assert 0
      if helix_score < min_perp_helix:
         return 9e9

      return 0.0

   return func

def make_const_jitfunc(v):
   @util.jit
   def dummy(*_):
      return v

   return dummy

@util.jit
def _get_coords_resbound(v, ibb, ih, reslb, resub, xseg, axis):
   begres = v.helixresbeg[ibb, ih]
   endres = v.helixresend[ibb, ih]
   beg = xseg @ v.helixbeg[ibb, ih, :]
   end = xseg @ v.helixend[ibb, ih, :]
   if reslb > begres or resub < endres:
      zbeg, zend = -12345.0, -12345.0
   else:
      zbeg = np.sum(axis * beg)
      zend = np.sum(axis * end)
   return beg, end, zbeg, zend

@util.jit
def _get_resbounds(v, vidx):
   reslb, resub = v.ires[vidx]
   if reslb == -1: reslb = -12345
   if resub == -1: resub = +12345
   if reslb > resub: resub, reslb = reslb, resub
   assert reslb < resub
   return reslb, resub

@util.jit
def _get_z_bounds(pos, idx, verts, axis, trim_bblocks, use_hull, zpad):
   zlb, zub = 9e9, -9e9
   for iv in range(len(verts) - trim_bblocks):
      v = verts[iv]
      xseg = pos[iv]
      vidx = idx[iv]
      ibb = v.ibblock[vidx]
      reslb, resub = _get_resbounds(v, vidx)
      if use_hull:
         for ihull in range(v.numhull[ibb]):
            pt = xseg @ v.hull[ibb, ihull]
            z = np.sum(axis * pt)
            zlb = min(z, zlb)
            zub = max(z, zub)
      else:
         for ih in range(v.numhelix[ibb]):
            _, _, zbeg, zend = _get_coords_resbound(v, ibb, ih, reslb, resub, xseg, axis)
            if zbeg == -12345.0: continue
            zlb = min(zbeg, zlb)
            zub = max(zbeg, zub)
            zlb = min(zend, zlb)
            zub = max(zend, zub)
   zlb += zpad
   zub -= zpad
   return zlb, zub

def _fixed_symaxis_check():
   axs, ang, cen = numba_axis_angle_cen(np.eye)
   return True
