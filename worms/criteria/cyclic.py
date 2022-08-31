from math import sqrt, acos

import willutil as wu
import worms

from worms.criteria.base import *
from worms import util
from worms.util import Bunch

# khash_cffi needs updating for numba 0.49+
# from worms.criteria import make_hash_table, WheelHashCriteria

import worms.homog as hm
from worms.homog import numba_axis_angle_cen, hrot
# from xbin import gu_xbin_indexer, numba_xbin_indexpyrosettaer
from copy import deepcopy
from worms.merge.concat import merge_results_concat

import warnings

def make_hash_table(*__arg__, **__kw__):
   raise NotImplementedError('khash_cffi needs updating for numba 0.49+')

def WheelHashCriteria(*__arg__, **__kw__):
   raise NotImplementedError('khash_cffi needs updating for numba 0.49+')

warnings.filterwarnings('ignore')

class Cyclic(WormCriteria):
   def __init__(
      self,
      symmetry=1,
      from_seg=0,
      *,
      tolerance=1.0,
      origin_seg=None,
      lever=50.0,
      to_seg=-1,
      min_radius=0,
      reference_structure=None,
      target_structure=None,
      fixori_segment=None,
      fixori_tolerance=10.0,
      **kw,
   ):
      super().__init__(**kw)
      if from_seg == to_seg:
         raise ValueError("from_seg should not be same as to_seg")
      if from_seg == origin_seg:
         raise ValueError("from_seg should not be same as origin_seg")
      if to_seg == origin_seg:
         raise ValueError("to_seg should not be same as origin_seg")
      if isinstance(symmetry, int):
         symmetry = "C" + str(symmetry)
      self.symmetry = symmetry
      self.tolerance = tolerance
      self.from_seg = from_seg
      self.origin_seg = origin_seg
      self.lever = lever
      self.to_seg = to_seg
      self.rot_tol = tolerance / lever
      # self.relweight = relweight if abs(relweight) > 0.001 else None
      if self.symmetry[0] in "cC":
         self.nfold = int(self.symmetry[1:])
         if self.nfold <= 0:
            raise ValueError("invalid symmetry: " + symmetry)
         self.symangle = np.pi * 2.0 / self.nfold
      else:
         raise ValueError("can only do Cx symmetry for now")
      if self.tolerance <= 0:
         raise ValueError("tolerance should be > 0")
      self.last_body_same_as = self.from_seg
      self.is_cyclic = True
      self.symname = None
      if self.nfold > 1:
         self.symname = "C" + str(self.nfold)
      self.sym_axes = [(self.nfold, Uz, [0, 0, 0, 1])]
      a = self.symangle

      self.min_radius = min_radius
      if self.nfold == 1:
         self.min_sep2 = 0.0
      elif self.nfold == 2:
         self.min_sep2 = 2.0 * min_radius
      else:
         self.min_sep2 = min_radius * np.sin(a) / np.sin((np.pi - a) / 2)
      self.min_sep2 = self.min_sep2**2

      self.fixori_segment = fixori_segment
      self.fixori_tolerance = fixori_tolerance
      self.fixori_target = np.eye(4)
      if self.fixori_segment is not None:
         raise NotImplementedError
         from worms.util.rosetta_utils import get_bb_stubs

         self.reference_structure = worms.rosetta_init.pose_from_file(reference_structure)
         self.target_structure = pose_from_file(target_structure)
         assert self.reference_structure.sequence() == self.target_structure.sequence()
         refstub = get_bb_stubs(self.reference_structure, which_resi=[7])[0].squeeze()
         tgtstub = get_bb_stubs(self.target_structure, which_resi=[7])[0].squeeze()
         self.fixori_target = tgtstub @ np.linalg.inv(refstub)

      self.bbspec = None  # should be filled in elsewhere

   def __eq__(self, other):
      return all([
         type(self) == type(other),
         self.symmetry == other.symmetry,
         self.tolerance == other.tolerance,
         self.from_seg == other.from_seg,
         self.origin_seg == other.origin_seg,
         self.lever == other.lever,
         self.to_seg == other.to_seg,
         self.min_radius == other.min_radius,
         self.nfold == other.nfold,
         self.min_sep2 == other.min_sep2,
         self.fixori_segment == other.fixori_segment,
         self.fixori_tolerance == other.fixori_tolerance,
         worms.util.generic_equals(self.fixori_target, other.fixori_target),
         self.bbspec == other.bbspec,
      ])

   def jit_lossfunc(self, **kw):
      kw = Bunch(**kw)

      tgt_ang = self.symangle
      from_seg = self.from_seg
      to_seg = self.to_seg
      lever = self.lever
      min_sep2 = self.min_sep2

      tolerance = float(kw.tolerance)  # type: ignore

      axis_constraint_bblock = -1
      axis_constraint_angle = 0.6523580032234328  # 37.37739188761675

      fixori_segment = self.fixori_segment
      fixori_target = self.fixori_target.astype(np.float32)
      fixori_tolerance = np.radians(self.fixori_tolerance)

      helixconf_filter = worms.filters.helixconf_jit.make_helixconf_filter(self, **kw)

      repeat_axis_filter = worms.filters.repeat_axis_jit.make_repeat_axis_filter_cyclic(
         self, **kw)

      @util.jit  # type: ignore
      def lossfunc(pos, idx, verts, debug=False):
         x_from = pos[from_seg]
         x_to = pos[to_seg]
         xhat = x_to @ np.linalg.inv(x_from)
         if debug: print('if np.sum(xhat[:3, 3]**2) < min_sep2:')
         if np.sum(xhat[:3, 3]**2) < min_sep2:
            return 9e9

         if debug: print('axis, angle, cen = numba_axis_angle_cen(xhat)')
         tmp = numba_axis_angle_cen(xhat)
         if tmp is None:
            return 9e9
         cycaxis, cycangle, _ = tmp

         rot_err_sq = lever**2 * (cycangle - tgt_ang)**2
         cart_err_sq = (np.sum(xhat[:, 3] * cycaxis))**2
         geomerr = np.sqrt(rot_err_sq + cart_err_sq)
         if geomerr > 10 * tolerance:
            return 9e9

         helixerr = helixconf_filter(pos, idx, verts, cycaxis)
         if helixerr > 100 * tolerance:
            return 9e9

         tgtaxiserr = 0
         fixorierr = 0
         if axis_constraint_bblock > 0:
            bbz = pos[axis_constraint_bblock, :, 2]
            bb_ang = acos(np.sum(bbz * cycaxis))
            bb_ang_err_sq = lever**2 * (bb_ang - axis_constraint_angle)**2
            # bb_pt = pos[axis_constraint_bblock, :, 3]

            if bb_ang_err_sq < 2.0:
               return 9e9

            tgtaxiserr = bb_ang_err_sq
            if debug: print('endif axis_constraint_bblock > 0:            ')

         if fixori_segment is not None:
            tgtpos = xhat @ fixori_target
            segpos = pos[fixori_segment] @ np.linalg.inv(x_from)
            tgtxform = tgtpos @ np.linalg.inv(segpos)
            # if debug: print(tgtxform)
            # if debug: print(tgtxform.shape)

            tgtaxis, _, _ = numba_axis_angle_cen(tgtxform, debug=debug)

            # if debug: print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # if debug: print('tgtaxis/ang/cen', cycaxis, tgtaxis)

            dev_angle = acos(np.abs(np.sum(cycaxis * tgtaxis)))
            dev_angle = min(dev_angle, np.pi - dev_angle)
            # if debug: print('dev_angle', dev_angle)
            fixorierr = np.sqrt(lever**2 * dev_angle**2)
            # if debug: print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            # assert 0

            # print('dev_angle', dev_angle, 'fixori_tolerance', fixori_tolerance)
            if dev_angle > fixori_tolerance:
               return 9e9
            # else:
            # return 0

         repeataxiserr = repeat_axis_filter(pos, idx, verts, cycaxis)
         if repeataxiserr > 10: return 9e9

         err = np.sqrt(geomerr**2 + helixerr**2 + tgtaxiserr**2 + fixorierr**2 + repeataxiserr**2)
         # if err < 5:
         # print('errors', err, 'geom', geomerr, 'hel', helixerr, 'tax', tgtaxiserr, 'ori',
         # fixorierr)

         return err

      @util.jit  #type:ignore
      def func1(pos, __idx__, __verts__):
         x_from = pos[from_seg]
         x_to = pos[to_seg]
         xhat = x_to @ np.linalg.inv(x_from)
         cosang = (xhat[0, 0] + xhat[1, 1] + xhat[2, 2] - 1.0) / 2.0
         rot_err_sq = (1.0 - cosang) * np.pi * lever**2  # hokey, but works
         # axis, angle = numba_axis_angle(xhat)  # right way, but slower
         # rot_err_sq = angle**2 * lever**2
         cart_err_sq = np.sum(xhat[:3, 3]**2)
         return np.sqrt(rot_err_sq + cart_err_sq)

      if self.nfold == 1:
         return func1

      return lossfunc

   def score(self, segpos, *, verbosity=False, **_):

      x_from = segpos[self.from_seg]
      x_to = segpos[self.to_seg]
      xhat = x_to @ inv(x_from)
      trans = xhat[..., :, 3]
      if self.nfold == 1:
         angle = hm.angle_of(xhat)
         carterrsq = np.sum(trans[..., :3]**2, axis=-1)
         roterrsq = angle**2
      else:
         if self.origin_seg is not None:
            tgtaxis = segpos[self.origin_seg] @ [0, 0, 1, 0]
            tgtcen = segpos[self.origin_seg] @ [0, 0, 0, 1]
            axis, angle, cen = hm.axis_ang_cen_of(xhat)
            carterrsq = hm.hnorm2(cen - tgtcen)
            roterrsq = (1 - np.abs(hm.hdot(axis, tgtaxis))) * np.pi
         else:  # much cheaper if cen not needed
            axis, angle = hm.axis_angle_of(xhat)
            carterrsq = roterrsq = 0
         carterrsq = carterrsq + hm.hdot(trans, axis)**2
         roterrsq = roterrsq + (angle - self.symangle)**2
         carterrsq[np.sum(trans[..., :3]**2, axis=-1) < self.min_sep2] = 9e9

         # if self.relweight is not None:
         #     # penalize 'relative' error
         #     distsq = np.sum(trans[..., :3]**2, axis=-1)
         #     relerrsq = carterrsq / distsq
         #     relerrsq[np.isnan(relerrsq)] = 9e9
         #     # too much of a hack??
         #     carterrsq += self.relweight * relerrsq
         if verbosity > 0:
            print("axis", axis[0])
            print("trans", trans[0])
            print("dot trans", hm.hdot(trans, axis)[0])
            print("angle", angle[0] * 180 / np.pi)

      return np.sqrt(carterrsq / self.tolerance**2 + roterrsq / self.rot_tol**2)

   def alignment(self, segpos, alignto='mid', **_):
      assert alignto in 'beg mid end'.split()
      if self.origin_seg is not None:
         return inv(segpos[self.origin_seg])
      if self.nfold == 1:
         return np.eye(4)
      x_from = segpos[self.from_seg]
      x_to = segpos[self.to_seg]
      xhat = x_to @ inv(x_from)
      axis, ang, cen = hm.axis_ang_cen_of(xhat)

      # print('aln', axis)
      # print('aln', ang * 180 / np.pi)
      # print('aln', cen)
      # print('aln', xhat[..., :, 3])
      dotz = hm.hdot(axis, Uz)[..., None]
      tgtaxis = np.where(dotz > 0, [0, 0, 1, 0], [0, 0, -1, 0])
      align = hm.hrot((axis + tgtaxis) / 2, np.pi, cen)
      align[..., :3, 3] -= cen[..., :3]

      alnx_from = align @ segpos[self.from_seg]
      alnx_to = align @ segpos[self.to_seg]
      alnxhat = alnx_to @ inv(alnx_from)  # close to ideal Cx rotation
      alnxhatinv = alnx_from @ inv(alnx_to)  # close to ideal Cx rotation
      xideal = hm.hrot([0, 0, 1], self.symangle)  # ideal Cx rotation
      xidealinv = hm.hrot([0, 0, 1], -self.symangle)  # ideal Cx rotation

      # print(alignto)
      # print(xideal)
      # print(alnxhat)

      # if alignto == 'beg':
      # align = alnxhatinv @ xidealinv @ align
      if alignto == 'end':
         align = alnxhatinv @ xidealinv @ align

      # print(alnxhat @ xideal)
      # print(xideal @ alnxhat)
      # assert 0

      return align

   def stages(self, hash_cart_resl, hash_ori_resl, bbs, **_):
      "return spearate criteria for each search stage"
      if self.origin_seg is None:
         return [(self, bbs)], None

      assert self.origin_seg == 0
      bbspec = deepcopy(self.bbspec[self.from_seg:])  # type:ignore
      bbspec[0][1] = "_" + bbspec[0][1][1]
      critA = Cyclic(
         self.nfold,
         min_radius=self.min_radius,
         lever=self.lever,
         tolerance=self.tolerance * 2.0,
      )
      critA.bbspec = bbspec
      bbsA = bbs[self.from_seg:] if bbs else None
      bbsB = bbs[:self.from_seg + 1] if bbs else None

      def stageB(__critA__, ssdagA, resultA):
         bbspec = deepcopy(self.bbspec[:self.from_seg + 1])  # type:ignore
         bbspec[-1][1] = bbspec[-1][1][0] + "_"
         gubinner = gu_xbin_indexer(hash_cart_resl, hash_ori_resl)
         numba_binner = numba_xbin_indexer(hash_cart_resl, hash_ori_resl)
         __keys__, hash_table = make_hash_table(ssdagA, resultA, gubinner)  # type:ignore
         critB = WheelHashCriteria(self, numba_binner, hash_table)
         critB.bbspec = bbspec  #type: ignore
         return critB

      return [(critA, bbsA), (stageB, bbsB)], merge_results_concat

   def merge_segment(self, **_):
      return self.from_seg

   def cloned_segments(self):
      "which bbs are being merged together"
      return self.from_seg, self.to_seg

   def iface_rms(self, pose0, prov, **_):
      return -1
      #       if self.origin_seg is None:
      #          # print('WARNING: iface_rms not implemented for simple cyclic')
      #          return -1
      #       else:
      #          same_as_last = list()
      #          for i, pr in enumerate(prov[:-1]):
      #             if pr[2] is prov[-1][2]:
      #                same_as_last.append(i)
      #          if len(same_as_last) < 2:
      #             print("iface_rms ERROR, not 3 merge subs! same_as_last:", same_as_last)
      #             # for i, (lb, ub, src, slb, sub) in enumerate(prov):
      #             # print(i, lb, ub, id(src), len(src), slb, sub)
      #             return 9e9

      #          i1, i2 = same_as_last[-2:]
      #          i3 = -1
      #          a1 = util.subpose(pose0, prov[i1][0], prov[i1][1])
      #          a2 = util.subpose(pose0, prov[i2][0], prov[i2][1])
      #          a3 = util.subpose(pose0, prov[i3][0], prov[i3][1])
      #          b1 = util.subpose(prov[i1][2], prov[i1][3], prov[i1][4])
      #          b2 = util.subpose(prov[i2][2], prov[i2][3], prov[i2][4])
      #          b3 = util.subpose(prov[i3][2], prov[i3][3], prov[i3][4])

      #          forward = hrot([0, 0, 1], 360.0 / self.nfold)
      #          backward = hrot([0, 0, 1], -720.0 / self.nfold)
      #          util.xform_pose(forward, a1)
      #          # a1.dump_pdb('a3_forward.pdb')
      #          fdist = a1.residue(1).xyz(2).distance(a3.residue(1).xyz(2))
      #          util.xform_pose(backward, a1)
      #          # a1.dump_pdb('a3_backward.pdb')
      #          bdist = a1.residue(1).xyz(2).distance(a3.residue(1).xyz(2))
      #          if bdist > fdist:
      #             util.xform_pose(forward, a1)
      #             util.xform_pose(forward, a1)

      #          # pose0.dump_pdb('pose0.pdb')
      #          # a1.dump_pdb('a1.pdb')
      #          # a2.dump_pdb('a2.pdb')
      #          # a3.dump_pdb('a3.pdb')
      #          # prov[-1][2].dump_pdb('src.pdb')
      #          # b1.dump_pdb('b1.pdb')
      #          # b2.dump_pdb('b2.pdb')
      #          # b3.dump_pdb('b3.pdb')

      #          ros.core.pose.append_pose_to_pose(a1, a2, True)
      #          ros.core.pose.append_pose_to_pose(a1, a3, True)
      #          ros.core.pose.append_pose_to_pose(b1, b2, True)
      #          ros.core.pose.append_pose_to_pose(b1, b3, True)

      #          # a1.dump_pdb('a.pdb')
      #          # b1.dump_pdb('b.pdb')

      #          rms = ros.core.scoring.CA_rmsd(a1, b1)
      #          # assert 0, 'debug iface rms ' + str(rms)

      #          return rms
