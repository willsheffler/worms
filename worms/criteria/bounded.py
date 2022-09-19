'''
this file contains logic for cage/dihedral and some crystals which include these
'''

from os import sep

from dask.base import register_pandas
from .base import WormCriteria, Ux, Uy, Uz
import numpy as np

import worms
import worms.homog as hm
from worms.homog import (
   numba_dihedral,
   numba_cross,
   numba_dot,
   numba_normalized,
   numba_line_line_closest_points_pa,
   numba_line_line_distance_pa,
)
from worms import PING
from worms.util import jit, generic_equals
from worms.merge.wye import wye_merge

DIHEDRAL_ROT_TOL = 1.0

class AxesIntersect(WormCriteria):
   """
    """
   def __init__(
      self,
      symname,
      tgtaxis1,
      tgtaxis2,
      from_seg=0,
      origin_seg=None,
      *,
      tolerance=1.0,
      lever=50,
      to_seg=-1,
      nondistinct_axes=False,
      segs=None,
      tgtaxis3=None,
      xtal=None,
      end_dihedral_sym_ang=None,
      **kw,
   ):
      """
      """
      super().__init__(**kw)
      if from_seg == to_seg:
         raise ValueError("from_seg should not be same as to_seg")
      self.symname = symname
      self.from_seg = from_seg
      if len(tgtaxis1) == 2:
         tgtaxis1 += ([0, 0, 0, 1], )
      if len(tgtaxis2) == 2:
         tgtaxis2 += ([0, 0, 0, 1], )
      self.tgtaxis1 = (
         tgtaxis1[0],
         hm.hnormalized(tgtaxis1[1]),
         hm.hpoint(tgtaxis1[2]),
      )
      self.tgtaxis2 = (
         tgtaxis2[0],
         hm.hnormalized(tgtaxis2[1]),
         hm.hpoint(tgtaxis2[2]),
      )
      if tgtaxis3:
         assert 0, 'tgtaxis3 needs an audit'
         if len(tgtaxis3) == 2:
            tgtaxis3 += ([0, 0, 0, 1], )
         self.tgtaxis3 = (
            tgtaxis3[0],
            hm.hnormalized(tgtaxis3[1]),
            hm.hpoint(tgtaxis3[2]),
         )
         assert 3 == len(self.tgtaxis3)
      assert 3 == len(self.tgtaxis1)
      assert 3 == len(self.tgtaxis2)

      self.tgtangle = hm.angle(tgtaxis1[1], tgtaxis2[1])
      self.tolerance = tolerance
      self.lever = lever
      self.to_seg = to_seg
      self.rot_tol = tolerance / lever
      self.nondistinct_axes = nondistinct_axes  # -z not same as z (for T33)
      # self.sym_axes = [self.tgtaxis1, self.tgtaxis2]
      self.is_cyclic = False
      self.origin_seg = None
      self.segs = segs
      self.xtal = xtal
      self.cell_dist_scale = 1.0
      self.end_dihedral_sym_ang = end_dihedral_sym_ang

   def __eq__(self, other):
      return all([
         type(self) == type(other),
         self.symname == other.symname,
         self.from_seg == other.from_seg,
         generic_equals(self.tgtaxis1, other.tgtaxis1),
         generic_equals(self.tgtaxis2, other.tgtaxis2),
         # self.tgtaxis1[0] == other.tgtaxis1[0],
         # self.tgtaxis2[0] == other.tgtaxis2[0],
         # np.allclose(self.tgtaxis1[1], other.tgtaxis1[1]),
         # np.allclose(self.tgtaxis1[1], other.tgtaxis1[1]),
         # np.allclose(self.tgtaxis2[2], other.tgtaxis2[2]),
         # np.allclose(self.tgtaxis2[2], other.tgtaxis2[2]),
         self.tgtangle == other.tgtangle,
         self.tolerance == other.tolerance,
         self.lever == other.lever,
         self.to_seg == other.to_seg,
         self.rot_tol == other.rot_tol,
         self.nondistinct_axes == other.nondistinct_axes,
         # generic_equals(self.sym_axes, other.sym_axes),
         self.is_cyclic == other.is_cyclic,
         self.origin_seg == other.origin_seg,
         self.segs == other.segs,
         self.xtal == other.xtal,
         self.cell_dist_scale == other.cell_dist_scale,
      ])

   def score(self, segpos, verbosity=False, **kw):
      cen1 = segpos[self.from_seg][..., :, 3]
      cen2 = segpos[self.to_seg][..., :, 3]
      ax1 = segpos[self.from_seg][..., :, 2]
      ax2 = segpos[self.to_seg][..., :, 2]
      if self.nondistinct_axes:
         p, q = hm.line_line_closest_points_pa(cen1, ax1, cen2, ax2)
         dist = hm.hnorm(p - q)
         cen = (p + q) / 2
         ax1c = hm.hnormalized(cen1 - cen)
         ax2c = hm.hnormalized(cen2 - cen)
         ax1 = np.where(hm.hdot(ax1, ax1c)[..., None] > 0, ax1, -ax1)
         ax2 = np.where(hm.hdot(ax2, ax2c)[..., None] > 0, ax2, -ax2)
         ang = np.arccos(hm.hdot(ax1, ax2))
      else:
         dist = hm.line_line_distance_pa(cen1, ax1, cen2, ax2)
         ang = np.arccos(np.abs(hm.hdot(ax1, ax2)))
      roterr2 = (ang - self.tgtangle)**2
      return np.sqrt(roterr2 / self.rot_tol**2 + (dist / self.tolerance)**2)

   def jit_lossfunc(self, **kw):
      from_seg = self.from_seg
      to_seg = self.to_seg
      tgtangle = self.tgtangle
      tolerance = self.tolerance
      rot_tol = self.rot_tol
      nondistinct_axes = self.nondistinct_axes

      # temporary hard coded stuff
      # NFOLD = 3
      # #
      # tgtdang = np.pi / NFOLD / 2
      # end_dihedral_sym_ang = 2 * np.pi / NFOLD
      end_dihedral_sym_ang = self.end_dihedral_sym_ang
      # # print('AxesIntersect.jit_lossfunc end_dihedral_sym_ang', end_dihedral_sym_ang)

      repeat_axis_filter = worms.filters.repeat_axis_jit.make_repeat_axis_filter_axesisect(
         self, **kw)

      @jit
      def func(pos, idx, verts):
         # print('bounded.py:jit_lossfunc begin')
         cen1 = pos[from_seg][:, 3]
         cen2 = pos[to_seg][:, 3]
         ax1 = pos[from_seg][:, 2]
         ax2 = pos[to_seg][:, 2]
         if nondistinct_axes:
            cen1 = cen1[:3]
            cen2 = cen2[:3]
            ax1 = ax1[:3]
            ax2 = ax2[:3]
            p, q = numba_line_line_closest_points_pa(cen1, ax1, cen2, ax2)
            dist = hm.np.sqrt(np.sum((p - q)**2))
            cen = (p + q) / 2
            ax1c = numba_normalized(cen1 - cen)
            ax2c = numba_normalized(cen2 - cen)
            if np.sum(ax1 * ax1c) < 0:
               ax1 = -ax1
            if np.sum(ax2 * ax2c) < 0:
               ax2 = -ax2
            ang = np.arccos(np.sum(ax1 * ax2))
         else:
            p, q = numba_line_line_closest_points_pa(cen1, ax1, cen2, ax2)
            # dist = numba_line_line_distance_pa(cen1, ax1, cen2, ax2)
            dist = np.linalg.norm(p - q)
            ang = np.arccos(np.abs(numba_dot(ax1, ax2)))
         roterr2 = (ang - tgtangle)**2

         geomscore = np.sqrt(roterr2 / rot_tol**2 + (dist / tolerance)**2)
         if geomscore > 2.0:
            # print('bounded.py:jit_lossfunc DONE geomscore > 0.5')
            return 9e9
         # print(abs(ang - tgtangle), dist)

         # xhat = pos[-1] @ np.linalg.inv(pos[0])
         # p, q = numba_line_line_closest_points_pa(cen1, ax1, cen2, ax2)

         cagecen = (p + q) / 2.0
         flip = numba_dot(pos[-1][:, 2], numba_normalized(pos[-1][:, 3] - cagecen))
         if flip < 0.0:
            # print('bounded.py:jit_lossfunc DONE flip < 0.0')
            return 9e9

         if end_dihedral_sym_ang is not None:
            # print('dist tol', dist, tolerance, np.linalg.norm(p - q))
            # daxis = xhat @ np.array([1, 0, 0, 0], dtype=np.float32)
            daxis = pos[-1][:, 0]  # x dihedral axis (x)
            # print('cagecen', cagecen)
            # print('ax1', ax1)
            # print('ax2', ax2)
            # print('daxis', daxis)
            # print('p3', cagecen + ax2)
            # print('p4', cagecen + ax2 + daxis)

            # print(
            #    np.degrees(
            #       numba_dihedral(
            #          np.array([0, 0, 1, 1]),
            #          np.array([0, 0, 0, 1]),
            #          np.array([1, 1, 1, 1]),
            # ]         np.array([2, 2, 1, 1]),
            #       )))
            # assert 0 # -180 -> 60

            dang = numba_dihedral(
               cagecen + ax1,
               cagecen,
               cagecen + ax2,
               cagecen + ax2 + daxis,
            )
            dang = dang % end_dihedral_sym_ang

            dangerr = min(
               abs(dang - end_dihedral_sym_ang - tgtdang),
               abs(dang - tgtdang),
               abs(dang + end_dihedral_sym_ang - tgtdang),
            )
            if dangerr > np.radians(DIHEDRAL_ROT_TOL):
               # print('bounded.py:jit_lossfunc DONE dangerr > tol')
               return 9e9
         # print('deg:', dang * 180 / np.pi, 'rad:', dang)

         # dihedral symcen1, center, symcen2, bblock_orig
         # print(np.sqrt(roterr2), dist)
         # print(np.sqrt(roterr2 / rot_tol**2 + (dist / tolerance)**2))
         # print(roterr2, rot_tol, dist, tolerance)
         # assert 0

         repeataxiserr = repeat_axis_filter(pos, idx, verts, ax1, ax2)
         if repeataxiserr > 10: return 9e9
         geomscore += repeataxiserr

         score = geomscore

         # if idx[0] == 102 and idx[1] == 7487:  # and idx[2] == 322:
         #    print(
         #       "bounded.py",
         #       idx,
         #       score,
         #       verts[0].ibblock.shape,
         #       verts[1].ibblock.shape,
         #       verts[2].ibblock.shape,
         #    )

         # print('bounded.py:jit_lossfunc DONE')
         return score

      return func

   def symfile_modifiers(self, segpos):
      if self.xtal:
         x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
         if cell_dist is None: return None
         return dict(scale_positions=cell_dist * self.cell_dist_scale)
      else:
         return dict()

   def crystinfo(self, segpos):
      # CRYST1   85.001   85.001   85.001  90.00  90.00  90.00 P 21 3
      if self.xtal is None: return None
      # print("hi")
      spacegroup = dict(I432='I 4 3 2', )
      # try:
      if True:
         xtal = spacegroup[self.xtal]
      # except Exception:/home/sheffler/for/zhe
      #    xtal = self.xtal

      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      if cell_dist is None: return None

      cell_dist = abs(cell_dist * self.cell_dist_scale)
      return cell_dist, cell_dist, cell_dist, 90, 90, 90, xtal

   def symops(self, segpos):
      '''
      used to check symmetric clashes in ouptup.py instead of 
      MakeLatticeMover, which is too flaky 
      '''
      # print('bounded.py:symops xtal:', self.xtal)

      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      if cell_dist is None: return None

      # # print(self.tgtaxis1)
      # # print(self.tgtaxis2)
      # symops = list()
      # nfold1, axis1, cen1 = self.tgtaxis1
      # nfold2, axis2, cen2 = self.tgtaxis2
      # assert np.allclose(cen1, [0, 0, 0, 1])
      # assert np.allclose(cen2, [0, 0, 0, 1])
      # assert not hasattr(self, 'tgtaxis3')

      # return [
      #    hm.hrot(axis1, np.pi * 2 / nfold1),
      #    hm.hrot(axis2, np.pi * 2 / nfold2),
      # ]

      # print('    cell dist', cell_dist)
      if self.xtal and self.xtal.replace(' ', '').upper() == 'I432':
         ops = (
            hm.hrot([1, 1, 1], 120),
            hm.hrot([1, 1, 1], 240),
            hm.hrot([1, 1, 0], 180),
            hm.hrot([1, 0, 0], 90),
            hm.hrot([1, 0, 0], 180),
            hm.hrot([1, 0, 0], 270),
            hm.hrot([1, -1, 0], 180, [cell_dist / 4] * 3),
         )
         # for op in ops:
         # print(op)
         return ops
      # assert 0
      return None

   def alignment(
      self,
      segpos,
      out_cell_spacing=False,
      debug=0,
      alignto='mid',
      **kw,
   ):
      if hm.angle_degrees(self.tgtaxis1[1], self.tgtaxis2[1]) < 0.1:
         PING('axes parallel??', flush=True)
         assert False
         # return np.eye(4)
         return None
      cen1 = segpos[self.from_seg][..., :, 3]
      cen2 = segpos[self.to_seg][..., :, 3]
      ax1 = segpos[self.from_seg][..., :, 2]
      ax2 = segpos[self.to_seg][..., :, 2]

      assert alignto in 'beg mid end'.split()

      # print(segpos)
      # print('========')
      # print('ang', np.degrees(hm.angle(ax1, ax2)))
      if not self.nondistinct_axes and hm.angle(ax1, ax2) > np.pi / 2:
         # print('swap')
         ax2 = -ax2
      # print('alignax1', ax1)
      # print('alignax2', ax2)
      # print('ang', hm.angle_degrees(ax1, ax2))
      # assert 0
      p, q = hm.line_line_closest_points_pa(cen1, ax1, cen2, ax2)
      cen = (p + q) / 2
      # print('p', p)
      # print('q', q)
      # print('dist', hm.line_line_distance_pa(cen1, ax1, cen2, ax2))
      # print('cendiff1', hm.hnormalized(cen1 - cen))
      # print('cendiff2', hm.hnormalized(cen2 - cen))
      # print('tgtaxis1', self.tgtaxis1[1])
      # print('tgtaxis2', self.tgtaxis2[1])
      xalign = hm.align_vectors(ax1, ax2, self.tgtaxis1[1], self.tgtaxis2[1], alignto=alignto)
      # print(xalign)
      # print('cen', cen)
      xalign[..., :3, 3] = -(xalign @ cen)[:3]
      # print(xalign)
      # print('newax', xalign @ ax1)
      # print('newax', xalign @ ax2)

      if debug:
         print(
            "angs",
            hm.angle_degrees(ax1, ax2),
            hm.angle_degrees(self.tgtaxis1[1], self.tgtaxis2[1]),
         )
         print("ax1", ax1)
         print("ax2", ax2)
         print("xax1", xalign @ ax1)
         print("tax1", self.tgtaxis1[1])
         print("xax2", xalign @ ax2)
         print("tax2", self.tgtaxis2[1])
         raise AssertionError
         # if not (np.allclose(xalign @ ax1, self.tgtaxis1[1], atol=1e-2) and
         #         np.allclose(xalign @ ax2, self.tgtaxis2[1], atol=1e-2)):
         #     print(hm.angle(self.tgtaxis1[1], self.tgtaxis2[1]))
         #     print(hm.angle(ax1, ax2))
         #     print(xalign @ ax1)
         #     print(self.tgtaxis1[1])
         #     print(xalign @ ax2)
         #     print(self.tgtaxis2[1])
         #     raise AssertionError('hm.align_vectors sucks')

      if out_cell_spacing:
         # cell spacing = dist to Dx origin * 2
         x, y, z, _ = xalign @ segpos[-1][:, 3]
         # TODO what is this? why 2.0?
         if abs(x - y) >= 2.0: return None, None
         if abs(y - z) >= 2.0: return None, None
         if abs(z - x) >= 2.0: return None, None
         cell_spacing = 4 * (x + y + z) / 3
         return xalign, cell_spacing

      assert xalign is not None

      return xalign

   def merge_segment(self, **kw):
      if self.origin_seg is None:
         return None
      return self.from_seg

   def stages(self, hash_cart_resl, hash_ori_resl, bbs, topology, **kw):
      "return spearate criteria for each search stage"
      if topology.is_linear():
         return [(self, bbs)], None

      # 3 component cage
      paths = topology.paths()
      assert len(paths) == 2
      segmap = {s: i for i, s in enumerate(self.segs)}
      axes = [self.tgtaxis1, self.tgtaxis2, self.tgtaxis3]
      from_seg = paths[0][0]
      to_segA = paths[0][-1]
      to_segB = paths[1][-1]

      critA = AxesIntersect(
         symname=self.symname,
         from_seg=from_seg,
         to_seg=to_segA,
         tgtaxis1=axes[segmap[from_seg]],
         tgtaxis2=axes[segmap[to_segA]],
      )
      critA.bbspec = [self.bbspec[i] for i in paths[0]]
      bbsA = [bbs[i] for i in paths[0]]

      critB = AxesIntersect(
         symname=self.symname,
         from_seg=from_seg,
         to_seg=to_segB,
         tgtaxis1=axes[segmap[from_seg]],
         tgtaxis2=axes[segmap[to_segB]],
      )
      critB.bbspec = [self.bbspec[i] for i in paths[1]]
      bbsB = [bbs[i] for i in paths[1]]

      print("3 comp cage stages!")

      return [(critA, bbsA), (critB, bbsB)], wye_merge

   # def cloned_segments(self):
   # "which bbs are being merged together"
   # return (self.from_seg, )

   def iface_rms(self, pose0, prov, **kw):
      return -1

def D2(c2=0, c2b=-1, **kw):
   return AxesIntersect("D2", (2, Uz), (2, Ux), c2, to_seg=c2b, **kw)

def D3(c3=0, c2=-1, **kw):
   return AxesIntersect("D3", (3, Uz), (2, Ux), c3, to_seg=c2, **kw)

def D4(c4=0, c2=-1, **kw):
   return AxesIntersect("D4", (4, Uz), (2, Ux), c4, to_seg=c2, **kw)

def D5(c5=0, c2=-1, **kw):
   return AxesIntersect("D5", (5, Uz), (2, Ux), c5, to_seg=c2, **kw)

def D6(c6=0, c2=-1, **kw):
   return AxesIntersect("D6", (6, Uz), (2, Ux), c6, to_seg=c2, **kw)

def D3_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D3", (2, Ux), (2, hm.hrot(Uz, 180 / 3)[0]), c2, to_seg=c2b,
                        nondistinct_axes=True, **kw)

def D4_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D4", (2, Ux), (2, hm.hrot(Uz, 180 / 4)[0]), c2, to_seg=c2b, **kw)

def D5_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D5", (2, Ux), (2, hm.hrot(Uz, 180 / 5)[0]), c2, to_seg=c2b,
                        nondistinct_axes=True, **kw)

def D6_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D6", (2, Ux), (2, hm.hrot(Uz, 180 / 6)[0]), c2, to_seg=c2b, **kw)

def D7_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D7", (2, Ux), (2, hm.hrot(Uz, 180 / 7)[0]), c2, to_seg=c2b,
                        nondistinct_axes=True, **kw)

def D8_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D8", (2, Ux), (2, hm.hrot(Uz, 180 / 8)[0]), c2, to_seg=c2b, **kw)

def D9_22(c2=0, c2b=-1, **kw):
   return AxesIntersect("D9", (2, Ux), (2, hm.hrot(Uz, 180 / 9)[0]), c2, to_seg=c2b,
                        nondistinct_axes=True, **kw)

def Tetrahedral(c3=None, c2=None, c3b=None, **kw):
   if 1 is not (c3b is None) + (c3 is None) + (c2 is None):
      raise ValueError("must specify exactly two of c3, c2, c3b")
   if c2 is None:
      from_seg, to_seg, nf1, nf2, ex = c3b, c3, 7, 3, 2
   if c3 is None:
      from_seg, to_seg, nf1, nf2, ex = c3b, c2, 7, 2, 3
   if c3b is None:
      from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 7
   return AxesIntersect(
      "T",
      from_seg=from_seg,
      to_seg=to_seg,
      tgtaxis1=(min(3, nf1), hm.sym.tetrahedral_axes[nf1]),
      tgtaxis2=(min(3, nf2), hm.sym.tetrahedral_axes[nf2]),
      nondistinct_axes=(nf1 == 7),
      **kw,
   )

def Octahedral(c4=None, c3=None, c2=None, **kw):
   if c4 is not None and c3 is not None and c2 is not None:
      return AxesIntersect(
         "O",
         segs=(c2, c3, c4),
         tgtaxis1=(2, hm.sym.octahedral_axes[2]),
         tgtaxis2=(3, hm.sym.octahedral_axes[3]),
         tgtaxis3=(4, hm.sym.octahedral_axes[4]),
      )
   if 1 is not (c4 is None) + (c3 is None) + (c2 is None):
      raise ValueError("must specify exactly two of c4, c3, c2")
   if c2 is None:
      from_seg, to_seg, nf1, nf2, ex = c4, c3, 4, 3, 2
   if c3 is None:
      from_seg, to_seg, nf1, nf2, ex = c4, c2, 4, 2, 3
   if c4 is None:
      from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 4
   return AxesIntersect("O", from_seg=from_seg, to_seg=to_seg,
                        tgtaxis1=(nf1, hm.sym.octahedral_axes[nf1]),
                        tgtaxis2=(nf2, hm.sym.octahedral_axes[nf2]), **kw)

def Icosahedral(c5=None, c3=None, c2=None, **kw):
   if 1 is not (c5 is None) + (c3 is None) + (c2 is None):
      raise ValueError("must specify exactly two of c5, c3, c2")
   if c2 is None:
      from_seg, to_seg, nf1, nf2, ex = c5, c3, 5, 3, 2
   if c3 is None:
      from_seg, to_seg, nf1, nf2, ex = c5, c2, 5, 2, 3
   if c5 is None:
      from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 5
   return AxesIntersect("I", from_seg=from_seg, to_seg=to_seg,
                        tgtaxis1=(nf1, hm.sym.icosahedral_axes[nf1]),
                        tgtaxis2=(nf2, hm.sym.icosahedral_axes[nf2]), **kw)
