from . import WormCriteria
import numpy as np
import homog as hm
from worms.util import jit

class DihedralCyclicLattice2D(WormCriteria):
   """
    cases
    D3 C3

    D2 C4
    D4 C2
    D4 C4

    D2 C3
    D2 C6
    D3 C2
    D3 C6
    D6 C2
    D6 C3
    """
   def __init__(
         self,
         symname,
         from_seg,
         d_nfold,
         c_nfold,
         *,
         tolerance=1.0,
         lever=50,
         to_seg=-1,
         space_group_str=None,
   ):
      assert from_seg == 0
      self.symname = symname
      self.from_seg = from_seg
      assert d_nfold, c_nfold in (
         (3, 3),  # P3m
         (2, 4),  # P4m
         (4, 2),
         (4, 4),
         (2, 3),  # P6m
         (2, 6),
         (3, 2),
         (3, 6),
         (6, 2),
         (6, 3))
      self.d_nfold = d_nfold
      self.c_nfold = c_nfold
      self.d_2folds = get_2folds(d_nfold)
      self.d_2diag = [[0, np.sqrt(2), np.sqrt(2)], [np.sqrt(2), 0, np.sqrt(2)],
                      [np.sqrt(2), np.sqrt(2), 0]]
      self.tolerance = tolerance
      self.lever = lever
      self.to_seg = to_seg
      self.space_group_str = space_group_str
      self.is_cyclic = False
      self.origin_seg = None

      print(f"DihedralCyclicLattice2D D{d_nfold} component 2folds:")
      for x in self.d_2folds:
         print("   ", x, hm.angle_degrees(x, self.d_2folds[0]))

   def score(self, segpos, **kw):
      raise NotImplementedError

   def jit_lossfunc(self):
      d_nfold = self.d_nfold
      d_2diag = np.array(self.d_2diag)
      c_nfold = self.c_nfold
      to_seg = self.to_seg
      tolerance = self.tolerance
      lever = self.lever
      d_2folds = self.d_2folds
      origin = np.array([0, 0, 0])

      @jit
      def lossfunc_Dx_Cx(pos, idx, vrts):
         ax = pos[to_seg, :3, 2]
         cn = pos[to_seg, :3, 3]
         mn = 9e9
         for i in range(d_2folds.shape[0]):
            d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
            mn = min(d, mn)
         carterr2 = mn**2  # close to intersecting d_2fold
         angerr2 = (np.arccos(abs(ax[2])) * lever)**2  # C symaxis on z
         return np.sqrt(carterr2 + angerr2)

      @jit
      def lossfunc_Dx_C2(pos, idx, vrts):
         ax = pos[to_seg, :3, 2]
         cn = pos[to_seg, :3, 3]

         if abs(ax[2]) > 0.5:
            # case: c2 on z
            mn = 9e9
            for i in range(d_2folds.shape[0]):
               d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
               mn = min(d, mn)
            carterr2 = mn**2
            angerr2 = (np.arccos(abs(ax[2])) * lever)**2
            return np.sqrt(carterr2 + angerr2)
         else:
            # case: C2 in plane of sym and perp to one of d_2folds axis
            dot = 9e9
            for i in range(d_2folds.shape[0]):
               dot = min(dot, abs(np.sum(d_2folds[i] * ax)))
            angerr2_a = (np.arcsin(dot) * lever)**2  # perp to d_2fold
            angerr2_b = (np.arcsin(ax[2]) * lever)**2  # axis in plane
            carterr2 = cn[2]**2  # center in plane
            return np.sqrt(angerr2_a + angerr2_b + carterr2)

      @jit
      def lossfunc_D2_Cx(pos, idx, vrts):
         ax = pos[to_seg, :3, 2]
         cn = pos[to_seg, :3, 3]
         mn, mni = 9e9, -1
         mxdot, mxdoti = 0, -1
         for i in range(d_2folds.shape[0]):
            d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
            if d < mn:
               mn, mni = d, i
            dot = abs(np.sum(d_2folds[i] * ax))
            if dot > mxdot:
               mxdot, mxdoti = dot, i
         if mxdoti == mni:
            # intersect same axis as parallel to, bad
            return 9e9
         carterr2 = mn**2
         angerr2 = (np.arccos(mxdot) * lever)**2
         return np.sqrt(carterr2 + angerr2)

      @jit
      def lossfunc_D2_C2_Z(pos, idx, vrts):
         ax = pos[to_seg, :3, 2]
         cn = pos[to_seg, :3, 3]
         mn = 9e9
         for i in range(d_2folds.shape[0]):
            d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2diag[i])
            dot = abs(np.sum(d_2folds[i] * ax))
            angerr2 = (np.arccos(dot) * lever)**2
            mn = min(mn, np.sqrt(d**2 + angerr2))
         return mn

      if d_nfold > 2 and c_nfold > 2:
         return lossfunc_Dx_Cx
      elif d_nfold > 2 and c_nfold == 2:
         return lossfunc_Dx_C2
      elif d_nfold == 2 and c_nfold > 2:
         return lossfunc_D2_Cx
      elif d_nfold == 2 and c_nfold == 2:
         return lossfunc_D2_C2_Z
      else:
         print("this should not happen")
      raise NotImplementedError

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      ax = segpos[self.to_seg, :3, 2]
      cn = segpos[self.to_seg, :3, 3]

      if self.d_nfold > 2 and self.c_nfold > 2:
         mn, mni = 9e9, -1
         for i, tf in enumerate(self.d_2folds):
            d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
            if d < mn:
               mn, mni = d, i
         p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
         spacing = np.linalg.norm(p + q) / 2
         xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])
      elif self.d_nfold > 2 and self.c_nfold == 2:

         if abs(ax[2]) > 0.5:
            # case: c2 on z pick d2 isects axis
            mn, mni = 9e9, -1
            for i, tf in enumerate(self.d_2folds):
               d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
               if d < mn:
                  mn, mni = d, i
            p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
            spacing = np.linalg.norm(p + q) / 2
            xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])
         else:
            # case: c2 prep to z, pick D2 perp to axis
            mn, mni = 9e9, -1
            for i, tf in enumerate(self.d_2folds):
               d = abs(np.sum(tf * ax))
               if d < mn:
                  mn, mni = d, i
            p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
            spacing = np.linalg.norm(p + q) / 2
            xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])

      elif self.d_nfold == 2 and self.c_nfold > 2:
         mn, mni = 9e9, -1
         mxdot, mxdoti = 0, -1
         for i, tf in enumerate(self.d_2folds):
            d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
            if d < mn:
               mn, mni = d, i
            dot = abs(np.sum(tf * ax))
            if dot > mxdot:
               mxdot, mxdoti = dot, i
         assert mni != mxdoti
         p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
         spacing = np.linalg.norm(p + q) / 2
         # assumes d_folds are X Y Z ax[argmin] selects correct one
         xalign = hm.align_vectors(self.d_2folds[mni], self.d_2folds[mxdoti], [1, 0, 0],
                                   [0, 0, 1])
         # print("cn", cn)
         # print("ax", ax)
         # print("isect", self.d_2folds[mni])
         # print("align", self.d_2folds[mxdoti])
         # print("align xform", xalign)
         # print("aligned ax", xalign[:3, :3] @ ax)
         # assert 0

      elif self.d_nfold == 2 and self.c_nfold == 2:
         mn, mni = 9e9, -1
         for i, tf in enumerate(self.d_2folds):
            d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], self.d_2diag[i])
            dot = abs(np.sum(tf * ax))
            angerr2 = (np.arccos(dot) * self.lever)**2
            err = np.sqrt(d**2 + angerr2)
            if err < mn:
               mn = err
               mni = i
         p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2diag[mni])
         spacing = np.linalg.norm(p + q) / 2 / np.sqrt(2)
         # assumes d_folds are X Y Z ax[argmin] selects correct one
         xalign = hm.align_vectors(self.d_2diag[mni], self.d_2folds[mni], [1, 1, 0], [0, 0, 1])
      else:
         raise NotImplementedError

      if out_cell_spacing:
         return xalign, spacing
      else:
         return xalign

   def symfile_modifiers(self, segpos):
      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      return dict(scale_positions=cell_dist)

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

def get_2folds(n):
   if n is 2:
      return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
   if n is 3:
      return np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]])
   if n is 4:
      return np.array([
         [1, 0, 0],
         [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
         [0, 1, 0],
         [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
      ])
   assert 0, ""

def P3m_D3_C3(d3=0, c3=-1):
   return DihedralCyclicLattice2D("P3m_D3_C3_3", d_nfold=3, c_nfold=3, from_seg=d3, to_seg=c3)

def P4m_D2_C4(d2=0, c4=-1):
   return DihedralCyclicLattice2D("P4m_D2_C4_4", d_nfold=2, c_nfold=4, from_seg=d2, to_seg=c4)

def P4m_D4_C2(d4=0, c2=-1):
   return DihedralCyclicLattice2D("P4m_D4_C2_3", d_nfold=4, c_nfold=2, from_seg=d4, to_seg=c2)

def P4m_D4_C4(d4=0, c4=-1):
   return DihedralCyclicLattice2D("P4m_D4_C4_3", d_nfold=4, c_nfold=4, from_seg=d4, to_seg=c4)

def P6m_D2_C3(d2=0, c3=-1):
   return DihedralCyclicLattice2D("P6m_D2_C3_4", d_nfold=2, c_nfold=3, from_seg=d2, to_seg=c3)

def P6m_D2_C6(d2=0, c6=-1):
   return DihedralCyclicLattice2D("P6m_D2_C6_3", d_nfold=2, c_nfold=6, from_seg=d2, to_seg=c6)

def P6m_D3_C2(d3=0, c2=-1):
   return DihedralCyclicLattice2D("P6m_D3_C2_4", d_nfold=3, c_nfold=2, from_seg=d3, to_seg=c2)

def P6m_D3_C6(d3=0, c6=-1):
   return DihedralCyclicLattice2D("P6m_D3_C6_3", d_nfold=3, c_nfold=6, from_seg=d3, to_seg=c6)

class DihedralCyclicLattice3D(WormCriteria):
   def __init__(
         self,
         symname,
         from_seg,
         d_nfold,
         c_nfold,
         c_tgt_axis_isect,
         aligners,
         *,
         tolerance=1.0,
         lever=200,
         to_seg=-1,
         space_group_str=None,
         to_origin=[0, 0, 0],
   ):
      assert from_seg == 0
      self.symname = symname
      self.from_seg = from_seg
      assert d_nfold, c_nfold in (
         (3, 3),  # P3m
         (2, 4),  # P4m
         (4, 2),
         (4, 4),
         (2, 3),  # P6m
         (2, 6),
         (3, 2),
         (3, 6),
         (6, 2),
         (6, 3),
      )
      self.d_nfold = d_nfold
      self.c_nfold = c_nfold
      c_tgt_axis_isect = np.array(c_tgt_axis_isect)
      self.c_tgt_axis_isect = c_tgt_axis_isect / np.sqrt(
         np.sum(c_tgt_axis_isect**2, axis=-1)[..., None])
      self.aligners = aligners

      self.tolerance = tolerance
      self.lever = lever
      self.to_seg = to_seg
      self.space_group_str = space_group_str
      self.is_cyclic = False
      self.origin_seg = None
      self.to_origin = np.array(to_origin)

   def score(self, segpos, **kw):
      raise NotImplementedError

   def jit_lossfunc(self):
      d_nfold = self.d_nfold
      c_nfold = self.c_nfold
      c_tgt_axis_isect = self.c_tgt_axis_isect
      to_seg = self.to_seg
      tolerance = self.tolerance
      lever = self.lever
      origin = np.array([0, 0, 0])

      @jit
      def lossfunc_Dx_Cx(pos, idx, vrts):
         ax = pos[to_seg, :3, 2]
         cn = pos[to_seg, :3, 3]
         mn = 9e9
         for i in range(len(c_tgt_axis_isect)):
            c_axis_isects = c_tgt_axis_isect[i, 0]
            c_tgt_axis = c_tgt_axis_isect[i, 1]
            d = hm.numba_line_line_distance_pa(cn, ax, origin, c_axis_isects)
            a = np.arccos(abs(np.sum(c_tgt_axis * ax)))
            mn = min(mn, np.sqrt(d**2 + (a * lever)**2))
         return mn

      if d_nfold > 2 and c_nfold > 2:
         return lossfunc_Dx_Cx

      raise NotImplementedError

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      ax = segpos[self.to_seg, :3, 2]
      cn = segpos[self.to_seg, :3, 3]
      origin = np.array([0, 0, 0])

      if self.d_nfold > 2 and self.c_nfold > 2:
         mn, mni = 9e9, None
         for i, _ in enumerate(self.c_tgt_axis_isect):
            c_axis_isects, c_tgt_axis = _
            d = hm.numba_line_line_distance_pa(cn, ax, origin, c_axis_isects)
            a = np.arccos(abs(np.sum(c_tgt_axis * ax)))
            err = np.sqrt(d**2 + (a * self.lever)**2)
            if err < mn:
               mn = err
               mni = i
         c_axis_isects, c_tgt_axis = self.c_tgt_axis_isect[mni]
         p, q = hm.line_line_closest_points_pa(cn, ax, origin, c_axis_isects)
         cell_spacing = np.linalg.norm(p + q) / np.sqrt(2)  # 2x from p+q

         xalign = np.eye(4)
         if np.sum(c_axis_isects * q) > 0:
            xalign = hm.hrot(hm.hcross(c_axis_isects, c_tgt_axis), 180) @ xalign
         xalign = self.aligners[mni] @ xalign

         xalign[:3, 3] = self.to_origin * cell_spacing

         d = np.linalg.norm(p - q)
         a = np.arccos(abs(np.sum(c_tgt_axis * ax)))
         carterr2 = d**2
         angerr2 = (a * self.lever)**2

         if np.sum(c_axis_isects * q) > 0:
            print()
            print("alignment", d, a * self.lever, self.lever)
            print("ax", ax, xalign[:3, :3] @ ax)
            print("cn", cn, xalign @ [cn[0], cn[1], cn[2], 1])
            print("isect", p)
            print("isect", q)
            print("cell_spacing", cell_spacing)
            print("xalign", hm.axis_angle_of(xalign))
            print(xalign)

      else:
         raise NotImplementedError

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
      return cell_dist, cell_dist, cell_dist, 90, 90, 90, self.space_group_str

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

def P432_D4_C3(d4=0, c3=-1):
   return DihedralCyclicLattice3D(
      "P432_D4_C3_2",
      d_nfold=4,
      c_nfold=3,
      c_tgt_axis_isect=[
         [[+1, +1, 0], [+1, +1, +1]],
         [[+1, +1, 0], [+1, +1, -1]],
         [[+1, -1, 0], [+1, -1, +1]],
         [[+1, -1, 0], [+1, -1, -1]],
         [[-1, +1, 0], [-1, +1, +1]],
         [[-1, +1, 0], [-1, +1, -1]],
         [[-1, -1, 0], [-1, -1, +1]],
         [[-1, -1, 0], [-1, -1, -1]],  # trips up align_vectors
      ],
      aligners=[
         np.eye(4),
         hm.hrot([1, 1, 0], 180),
         hm.hrot([0, 0, 1], 90),
         hm.hrot([1, 1, 0], 180) @ hm.hrot([0, 0, 1], 90),
         hm.hrot([0, 0, 1], -90),
         hm.hrot([1, 1, 0], 180) @ hm.hrot([0, 0, 1], -90),
         hm.hrot([0, 0, 1], 180),
         hm.hrot([1, 1, 0], 180) @ hm.hrot([0, 0, 1], 180),
      ],
      from_seg=d4,
      to_seg=c3,
      to_origin=[0.5, 0.5, 0],
      space_group_str="P 4 3 2",
   )

#### from ariel

def P2m_D2_C2(d2=0, c2=-1):
   return DihedralCyclicLattice2D("P2m_D2_C2_4", d_nfold=2, c_nfold=2, from_seg=d2, to_seg=c2)
