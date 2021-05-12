from . import WormCriteria, Ux, Uz
import numpy as np
import homog as hm  ## python library that Will wrote to do geometry things
from worms.util import jit

class AxesAngle(WormCriteria):  ## for 2D arrays (maybe 3D in the future?)
   def __init__(
      self,
      symname,
      tgtaxis1,
      tgtaxis2,
      from_seg,
      *,
      tolerance=1.0,
      lever=50,
      to_seg=-1,
      space_group_str=None,
      cell_dist_scale=1.0,
      tgtaxis2_isects=[0, 1, 0],
   ):
      """ Worms criteria for non-intersecting axes re: unbounded things

        assume tgtaxis1 goes through origin
        tgtaxis2 intersects tgtaxis2_isects

        Args:
            symname (str): Symmetry identifier, to label stuff and look up the symdef file.
            tgtaxis1: Target axis 1.
            tgtaxis2: Target axis 2.
            from_seg (int): The segment # to start at.
            tolerance (float): A geometry/alignment error threshold. Vaguely Angstroms.
            lever (float): Tradeoff with distances and angles for a lever-like object. To convert an angle error to a distance error for an oblong shape.
            to_seg (int): The segment # to end at.
            space_group_str: The target space group.

        """

      self.symname = symname
      self.cell_dist_scale = cell_dist_scale
      self.tgtaxis1 = np.asarray(
         tgtaxis1, dtype="f8"
      )  ## we are treating these as vectors for now, make it an array if it isn't yet, set array type to 8-type float
      self.tgtaxis2 = np.asarray(tgtaxis2, dtype="f8")
      # print(self.tgtaxis1.shape)
      # print(np.linalg.norm(tgtaxis1))
      self.tgtaxis1 /= np.linalg.norm(self.tgtaxis1)  # normalize target axes to 1,1,1
      self.tgtaxis2 /= np.linalg.norm(self.tgtaxis2)
      if hm.angle(self.tgtaxis1, self.tgtaxis2) > np.pi / 2:
         self.tgtaxis2 = -self.tgtaxis2
      self.from_seg = from_seg
      self.tolerance = tolerance
      self.lever = lever
      self.to_seg = to_seg
      self.space_group_str = space_group_str
      ## if you want to store arguments, you have to write these self.argument lines

      self.target_angle = np.arccos(np.abs(hm.hdot(
         self.tgtaxis1, self.tgtaxis2)))  ## already set to a non- self.argument in this function
      # print(self.target_angle * (180 / np.pi))
      self.is_cyclic = False
      self.origin_seg = None
      self.tgtaxis2_isects = tgtaxis2_isects

   def score(self, segpos, **kw):
      ax1 = segpos[self.from_seg][..., :, 2]
      ax2 = segpos[self.to_seg][..., :, 2]
      angle = np.arccos(np.abs(hm.hdot(ax1, ax2)))
      return np.abs((angle - self.target_angle)) / self.tolerance * self.lever

   def jit_lossfunc(self):
      from_seg = self.from_seg
      to_seg = self.to_seg
      target_angle = self.target_angle
      tolerance = self.tolerance
      lever = self.lever

      @jit
      def func(pos, idx, verts):
         ax1 = pos[from_seg, :3, 2]
         ax2 = pos[to_seg, :3, 2]
         angle = np.arccos(np.abs(np.sum(ax1 * ax2)))
         return np.abs((angle - target_angle)) / tolerance * lever

      return func

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      """ Alignment to move stuff to be in line with symdef file

        Args:
            segpos (lst): List of segment positions / coordinates.
            **kw I'll accept any "non-positional" argument as name = value, and store in a dictionary

        """
      cen1 = segpos[self.from_seg][..., :, 3]
      cen2 = segpos[self.to_seg][..., :, 3]
      ax1 = segpos[self.from_seg][..., :, 2]  ## 3rd column is axis
      ax2 = segpos[self.to_seg][..., :, 2]
      ## make sure to align with smaller axis choice
      if hm.angle(ax1, ax2) > np.pi / 2:
         ax2 = -ax2
      if abs(hm.angle(self.tgtaxis1, self.tgtaxis2)) < 0.1:
         # vector delta between cen2 and cen1
         d = hm.proj_perp(ax1, cen2 - cen1)
         Xalign = hm.align_vectors(ax1, d, self.tgtaxis1,
                                   self.tgtaxis2_isects + [0])  # align d to Y axis
         Xalign[..., :, 3] = -Xalign @ cen1
         cell_dist = (Xalign @ cen2)[..., 1]
      else:
         try:
            Xalign = hm.align_vectors(ax1, ax2, self.tgtaxis1, self.tgtaxis2)
         except AssertionError as e:
            print("align_vectors error")
            print("   ", ax1)
            print("   ", ax2)
            print("   ", self.tgtaxis1)
            print("   ", self.tgtaxis2)
            raise e
         Xalign[..., :, 3] = -Xalign @ cen1  ## move from_seg cen1 to origin
         cen2_0 = Xalign @ cen2  # moving cen2 by Xalign
         D = np.stack([self.tgtaxis1[:3], self.tgtaxis2_isects, self.tgtaxis2[:3]]).T
         A1offset, cell_dist, _ = np.linalg.inv(D) @ cen2_0[:3]
         # transform of A1 offest, cell distance (offset along other axis), and A2 offset (<-- we are ignoring this)
         Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * self.tgtaxis1)
         # Xalign[..., :, 3] = Xalign[..., :, 3] + [0,cell_dist,0,0]
      if out_cell_spacing:
         return Xalign, cell_dist
      else:
         return Xalign

   def symfile_modifiers(self, segpos):
      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      return dict(scale_positions=cell_dist * self.cell_dist_scale)

   def crystinfo(self, segpos):
      # CRYST1   85.001   85.001   85.001  90.00  90.00  90.00 P 21 3
      if self.space_group_str is None:
         return None
      # print("hi")
      x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
      cell_dist = abs(cell_dist * self.cell_dist_scale)
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

def Sheet_P321(c3=None, c2=None, **kw):
   if c3 is None or c2 is None:
      raise ValueError("must specify ...?")
   return AxesAngle("Sheet_P321_C3_C2_depth3_1comp", Uz, Ux, from_seg=c3, to_seg=c2, **kw)

##this is currently identical to the D3 format...how do we change it to make it an array?

def Sheet_P4212(c4=None, c2=None, **kw):
   ##should there be options for multiple C2's?
   if c4 is None or c2 is None:
      raise ValueError("must specify ...?")  # one or two of c4, c2

#"Sheet_P4212_C4_C2_depth3_1comp",
   return AxesAngle('P4212_C4_C2_6', Uz, Ux, from_seg=c4, to_seg=c2, cell_dist_scale=2.0, **kw)

def Sheet_P6(c6=None, c2=None, **kw):
   if c6 is None or c2 is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   return AxesAngle("Sheet_P6_C6_C2_depth3_1comp", Uz, Uz, from_seg=c6, to_seg=c2, **kw)

def Sheet_P6_C3_C2(c3=None, c2=None, **kw):
   if c3 is None or c2 is None:
      raise ValueError("must specify ...?")
   return AxesAngle("Sheet_P6_C6_C2_depth3_1comp", Uz, Uz, from_seg=c3, to_seg=c2, **kw)

#### WORKING ####
def Crystal_P213_C3_C3(c3a=None, c3b=None, **kw):
   if c3a is None or c3b is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_P213_C3_C3_depth3_1comp",
      [1, 1, 1, 0],
      [-1, -1, 1, 0],
      from_seg=c3a,
      to_seg=c3b,
      space_group_str="P 21 3",
      cell_dist_scale=2.0,  # for some reason, this one needs this
      **kw,
   )
   # dihedral angle = 70.5288

#### IN PROGRESS ####
# I just normalized all the angles, but I don't think you can do this...might need to check the angle between them. Print and check that it is correct.
def Crystal_P4132_C2_C3(c2a=None, c3b=None, **kw):
   if c3a is None or c3b is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_P4132_C2_C3_depth3_1comp",
      [0, -1, 1, 0],
      [-1, -1, 0, 0],
      from_seg=c2a,
      to_seg=c3b,
      space_group_str="P 41 3 2",
      **kw,
   )
   # dihedral angle = 35.2644

def Crystal_I213_C2_C3(c2a=None, c3b=None, **kw):
   if c3a is None or c3b is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_I213_C2_C3_depth3_1comp",
      [0, 0, 1, 0],
      [-1, 1, 1, 0],
      from_seg=c2a,
      to_seg=c3b,
      space_group_str="I 21 3",
      **kw,
   )
   # dihedral angle = 54.7356

def Crystal_I432_C2_C4(c2a=None, c4b=None, **kw):
   if c2a is None or c4b is None:
      raise ValueError("must specify ...?")
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_I432_C2_C4_depth3_1comp",
      [-1, 0, 1, 0],
      [0, 0, 1, 0],
      from_seg=c2a,
      to_seg=c4b,
      space_group_str="I 4 3 2",
      **kw,
   )
   # dihedral angle = 45

def Crystal_F432_C3_C4(c3a=None, c4b=None, **kw):
   if c3a is None or c3b is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_F432_C3_C4_depth3_1comp",
      [-1, 1, 1, 0],
      [0, 1, 0, 0],
      from_seg=c3a,
      to_seg=c4b,
      space_group_str="F 4 3 2",
      **kw,
   )
   # dihedral angle = 54.7356

def Crystal_P432_C4_C4(c4a=None, c4b=None, **kw):
   if c3a is None or c3b is None:
      raise ValueError("must specify ...?")  # one or two of c6, c2
   # return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
   return AxesAngle(
      "Crystal_P432_C4_C4_depth3_1comp",
      [0, 0, 1, 0],
      [0, 1, 0, 0],
      from_seg=c4a,
      to_seg=c4b,
      space_group_str="P 4 3 2",
      **kw,
   )
   # dihedral angle = 90

def Sheet_P42_from_ws_0127(c4=None, c2=None, **kw):
   ##should there be options for multiple C2's?
   if c4 is None or c2 is None:
      raise ValueError("must specify ...?")  # one or two of c4, c2
   return AxesAngle(
      "P42_C4_C2_4",
      Uz,
      Uz,
      from_seg=c4,
      to_seg=c2,
      cell_dist_scale=2.0,
      **kw,
   )

def Sheet_P6_C3_C2_from_ws_0202(c3=None, c2=None, **kw):
   if c3 is None or c2 is None:
      raise ValueError("must specify ...?")
   return AxesAngle(
      "P6_C3_C2_6",
      Uz,
      Uz,
      from_seg=c3,
      to_seg=c2,
      cell_dist_scale=2,
      **kw,
   )

def Sheet_P4212_from_ws_0127(c4=None, c2=None, **kw):
   ##should there be options for multiple C2's?
   if c4 is None or c2 is None:
      raise ValueError("must specify ...?")  # one or two of c4, c2
   return AxesAngle(
      "P4212_C4_C2_4",
      Uz,
      Ux,
      from_seg=c4,
      to_seg=c2,
      cell_dist_scale=2.0,
      **kw,
   )

class Crystal_F23_T_C2(AxesAngle):
   def __init__(self, t=None, c2=None, **kw):
      if t is None or c2 is None:
         raise ValueError("must specify ...?")
      super().__init__(
         "F23_T_C2_depth2_1comp",
         tgtaxis1=[0, 0, 1, 0],
         tgtaxis2=[0, 0, 1, 0],
         from_seg=t,
         to_seg=c2,
         space_group_str="F 2 3",
         cell_dist_scale=4.0,
         **kw,
      )

   def jit_lossfunc(self):
      from_seg = self.from_seg
      to_seg = self.to_seg
      target_angle = self.target_angle
      tolerance = self.tolerance
      lever = self.lever

      @jit
      def func(pos, idx, verts):

         cen2x = pos[to_seg, 0, 3]
         cen2y = pos[to_seg, 1, 3]
         dis_err2 = (cen2x - cen2y)**2

         ax1 = pos[from_seg, :3, 2]
         ax2 = pos[to_seg, :3, 2]
         angle = np.arccos(np.abs(np.sum(ax1 * ax2)))
         ang_err2 = (angle - target_angle)**2

         return np.sqrt(ang_err2 * lever**2 + dis_err2) / tolerance

      return func

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      Xalign = np.eye(4)
      cell_dist = segpos[-1][0, 3]  # x-coord

      if out_cell_spacing:
         return Xalign, cell_dist
      else:
         return Xalign

class Crystal_F23_T_C3(AxesAngle):
   def __init__(self, t=None, c3=None, **kw):
      if t is None or c3 is None:
         raise ValueError("must specify ...?")
      super().__init__(
         "F23_T_C3_depth2_1comp",
         tgtaxis1=[0, 0, 1, 0],
         tgtaxis2=[0, 0, 1, 0],
         from_seg=t,
         to_seg=c3,
         space_group_str="F 2 3",
         cell_dist_scale=1.0,
         **kw,
      )

   def jit_lossfunc(self):
      from_seg = self.from_seg
      to_seg = self.to_seg
      target_angle = self.target_angle
      tolerance = self.tolerance
      lever = self.lever
      centgt1 = np.array([2, 1, 1]) / np.sqrt(6)
      centgt2 = np.array([1, 2, 1]) / np.sqrt(6)
      centgt3 = np.array([1, 1, 2]) / np.sqrt(6)
      axstgt1 = np.array([-1, +1, +1]) / np.sqrt(3)
      axstgt2 = np.array([+1, -1, +1]) / np.sqrt(3)
      axstgt3 = np.array([+1, +1, -1]) / np.sqrt(3)

      @jit
      def func(pos, idx, verts):
         cen = pos[to_seg, :3, 3]
         axs = pos[to_seg, :3, 2]

         # axis ~= -1,1,1
         # cen along 2,1,1

         # axs norm already 1
         cenperp = cen - np.sum(axs * cen) * axs
         cenperp /= np.sqrt(np.sum(cenperp * cenperp))
         assert cenperp.shape == (3, )

         err1 = 1.0 - np.abs(np.sum(centgt1 * cenperp))
         err1 += 1.0 - np.abs(np.sum(axstgt1 * axs))
         # err2 = 1.0 - np.abs(np.sum(centgt2 * cenperp))
         # err2 += 1.0 - np.abs(np.sum(axstgt2 * axs))
         # err3 = 1.0 - np.abs(np.sum(centgt3 * cenperp))
         # err3 += 1.0 - np.abs(np.sum(axstgt3 * axs))
         # err = np.min(np.array([err1, err2, err3]))

         return np.sqrt(err1) * lever

      return func

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      Xalign = np.eye(4)

      cen = segpos[-1][:3, 3]
      axs = segpos[-1][:3, 2]
      cenperp = cen - np.sum(axs * cen) * axs
      cell_dist = cenperp[0] * 3

      if out_cell_spacing:
         return Xalign, cell_dist
      else:
         return Xalign

class Crystal_F23_T_T(AxesAngle):
   def __init__(self, t=None, tb=None, **kw):
      if t is None or tb is None:
         raise ValueError("must specify ...?")
      super().__init__(
         "F23_T_T_depth2_1comp",
         tgtaxis1=[0, 0, 1, 0],
         tgtaxis2=[0, 0, 1, 0],
         from_seg=t,
         to_seg=tb,
         space_group_str="F 2 3",
         cell_dist_scale=2.0,
         **kw,
      )

   def jit_lossfunc(self):
      from_seg = self.from_seg
      to_seg = self.to_seg
      target_angle = self.target_angle
      tolerance = self.tolerance
      lever = self.lever

      @jit
      def func(pos, idx, verts):

         cen2x = pos[to_seg, 0, 3]
         cen2y = pos[to_seg, 1, 3]
         cen2z = pos[to_seg, 2, 3]
         dis_err2 = (cen2x - cen2y)**2
         dis_err2 += (cen2x - cen2z)**2
         dis_err2 += (cen2y - cen2z)**2

         ax1 = pos[from_seg, :3, 2]
         ax2 = pos[to_seg, :3, 2]
         angle = np.arccos(np.abs(np.sum(ax1 * ax2)))
         ang_err2 = (angle - target_angle)**2

         return np.sqrt(ang_err2 * lever**2 + dis_err2) / tolerance

      return func

   def alignment(self, segpos, out_cell_spacing=False, **kw):
      Xalign = np.eye(4)
      cell_dist = segpos[-1][0, 3]  # x-coord

      if out_cell_spacing:
         return Xalign, cell_dist
      else:
         return Xalign
