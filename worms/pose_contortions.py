"""TODO: Summary
"""

import os
import itertools as it
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np

from worms import util

try:
   import pyrosetta
   from pyrosetta import rosetta as ros
   from pyrosetta import rosetta as ros

   rm_lower_t = ros.core.pose.remove_lower_terminus_type_from_pose_residue
   rm_upper_t = ros.core.pose.remove_upper_terminus_type_from_pose_residue
except ImportError:
   pass

class AnnoPose:
   def __init__(self, pose, iseg, srcpose, src_lb, src_ub, cyclic_entry):
      self.pose = pose
      self.iseg = iseg
      self.srcpose = srcpose
      self.src_lb = src_lb
      self.src_ub = src_ub
      self.cyclic_entry = cyclic_entry

   def __iter__(self):
      'this seems sketchy'
      yield self.pose
      yield (self.iseg, self.srcpose, self.src_lb, self.src_ub)

   def __getitem__(self, i):
      if i is 0:
         return self.pose
      if i is 1:
         return (self.iseg, self.srcpose, self.src_lb, self.src_ub)

   def seq(self):
      return self.pose.sequence()

   def srcseq(self):
      return self.srcpose.sequence()[self.src_lb - 1:self.src_ub]

CyclicTrim = namedtuple("CyclicTrim", "sym_seg_from sym_seg_to".split())

def contort_pose_chains(
      pose,
      chains,
      nseg,
      ir_en,
      ir_ex,
      pl_en,
      pl_ex,
      chain_start,
      chain_end,
      position=None,
      pad=(0, 0),
      iseg=None,
      cyclictrim=None,
      last_seg_entrypol=None,
      first_seg_exitpol=None,
      sym_ir=None,
      sym_pol=None,
):
   """make pose chains from 'segment' info

    what a monster this has become. returns (segchains, rest)
    segchains elems are [enterexitchain] or, [enterchain, ..., exitchain]
    rest holds other chains IFF enter and exit in same chain
    each element is a pair [pose, source] where source is
    (origin_pose, start_res, stop_res)
    cyclictrim specifies segments which are spliced across the
    symmetric interface. segments only needed if cyclictrim==True
    if cyclictrim, last segment will only be a single entry residue

    Args: No
    """
   cyclic_entry = defaultdict(lambda: None)
   if cyclictrim and cyclictrim[1] < 0:
      cyclictrim = cyclictrim[0], cyclictrim[1] + nseg

   if cyclictrim and iseg == cyclictrim[0]:
      # assert ir_en == -1, 'paece sign not implemented yet'
      sym_ch = pose.chain(sym_ir)
      cyclictrim_in_rest = False
      if ir_en == -1:
         ir_en = sym_ir
      else:
         entry_chain, exit_chain = pose.chain(ir_en), pose.chain(ir_ex)
         if sym_ch not in (entry_chain, exit_chain):
            cyclictrim_in_rest = True
         assert sym_ch != entry_chain or sym_pol != pl_en
         assert sym_ch != exit_chain or sym_pol != pl_ex
      # annotate enex entries with cyclictrim info
      cyclic_entry[pose.chain(sym_ir)] = (iseg, sym_ir, sym_pol)
   if cyclictrim and iseg == cyclictrim[1]:
      assert ir_ex == -1
      assert iseg + 1 == nseg
      i = ir_en
      p = util.subpose(pose, i, i)
      if position is not None:
         util.xform_pose(position, p)
      return [AnnoPose(p, iseg, pose, i, i, None)], []

   ch_en = pose.chain(ir_en) if ir_en > 0 else None
   ch_ex = pose.chain(ir_ex) if ir_ex > 0 else None

   if cyclictrim and iseg == 0:
      pl_en = last_seg_entrypol
   if cyclictrim and iseg + 1 == nseg:
      assert 0, "last segment not returned as single entry residue?!?"
      pl_ex = first_seg_exitpol
   if ch_en:
      ir_en -= chain_start[ch_en]
   if ch_ex:
      ir_ex -= chain_start[ch_ex]
   assert ch_en or ch_ex

   rest = OrderedDict()
   did_cyclictrim_in_rest = False
   for i in range(1, len(chains) + 1):
      pchain = chains[i]
      lb = chain_start[i] + 1
      ub = chain_end[i]
      if cyclic_entry[i] is not None:
         if i not in (ch_en, ch_ex):
            did_cyclictrim_in_rest = True
         ir = cyclic_entry[i][1] - chain_start[i]
         pchain, lb, ub = util.trim_pose(pchain, ir, cyclic_entry[i][2])
         lb += chain_start[i]
         ub += chain_start[i]
      rest[chains[i]] = AnnoPose(pchain, iseg, pose, lb, ub, cyclic_entry[i])
      assert rest[chains[i]].seq() == rest[chains[i]].srcseq()
   if cyclictrim and iseg == cyclictrim[0]:
      if cyclictrim_in_rest != did_cyclictrim_in_rest:
         print(
            "cyclictrim_in_rest",
            cyclictrim_in_rest,
            "did_cyclictrim_in_rest",
            did_cyclictrim_in_rest,
         )
         print("iseg", iseg, "len(chains)", len(chains))
         assert cyclictrim_in_rest == did_cyclictrim_in_rest
   if ch_en:
      del rest[chains[ch_en]]
   if ch_en == ch_ex:
      assert len(rest) + 1 == len(chains)
      p, l1, u1 = util.trim_pose(chains[ch_en], ir_en, pl_en, pad[0])
      iexit1 = ir_ex - (pl_ex == "C") * (len(chains[ch_en]) - len(p))
      p, l2, u2 = util.trim_pose(p, iexit1, pl_ex, pad[1] - 1)
      lb = l1 + l2 - 1 + chain_start[ch_en]
      ub = l1 + u2 - 1 + chain_start[ch_en]
      enex = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_en])]
      assert p.sequence() == pose.sequence()[lb - 1:ub]
      rest = list(rest.values())
   else:
      if ch_ex:
         del rest[chains[ch_ex]]
      p_en = [chains[ch_en]] if ch_en else []
      p_ex = [chains[ch_ex]] if ch_ex else []
      if p_en:
         p, lben, uben = util.trim_pose(p_en[0], ir_en, pl_en, pad[0])
         lb = lben + chain_start[ch_en]
         ub = uben + chain_start[ch_en]
         p_en = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_en])]
         assert p.sequence() == pose.sequence()[lb - 1:ub]
      if p_ex:
         p, lbex, ubex = util.trim_pose(p_ex[0], ir_ex, pl_ex, pad[1] - 1)
         lb = lbex + chain_start[ch_ex]
         ub = ubex + chain_start[ch_ex]
         p_ex = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_ex])]
         assert p.sequence() == pose.sequence()[lb - 1:ub]
      enex = p_en + list(rest.values()) + p_ex
      rest = []
   for ap in rest:
      s1 = str(ap.pose.sequence())
      s2 = str(ap.srcpose.sequence()[ap.src_lb - 1:ap.src_ub])
      if s1 != s2:
         print('WARNING: sequence mismatch in "rest", maybe OK, but '
               "proceed with caution and tell will to fix!")
         # print(s1)
         # print(s2)
      assert s1 == s2
   if position is not None:
      position = util.rosetta_stub_from_numpy_stub(position)
      for x in enex:
         x.pose = x.pose.clone()
      for x in rest:
         x.pose = x.pose.clone()
      for ap in it.chain(enex, rest):
         ros.protocols.sic_dock.xform_pose(ap.pose, position)
   for iap, ap in enumerate(it.chain(enex, rest)):
      assert isinstance(ap, AnnoPose)
      assert ap.iseg == iseg
      assert ap.seq() == ap.srcseq()
      # a = ap.seq()
      # b = ap.srcseq()
      # if a != b:
      # print('WARNING sequence mismatch!', iap, len(enex), len(rest))
      # print(a)
      # print(b)
      # assert a == b
   # print(iseg, 'enex', len(enex), 'unconnected', len(rest))
   # for i, ap in enumerate(enex):
   #    import os
   #    assert not os.path.exists(f'seg{iseg}_enex{i}.pdb')
   #    ap.pose.dump_pdb(f'seg{iseg}_enex{i}.pdb')

   return enex, rest

def reorder_spliced_as_N_to_C(body_chains, polarities):
   """remap chains of each body such that concatenated chains are N->C

    Args:
        body_chains (TYPE): Description
        polarities (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
   if len(body_chains) != len(polarities) + 1:
      raise ValueError("must be one more body_chains than polarities")
   chains, pol = [[]], {}
   if not all(0 < len(dg) for dg in body_chains):
      raise ValueError("body_chains values must be [enterexit], "
                       "[enter,exit], or [enter, ..., exit")
   for i in range(1, len(polarities)):
      if len(body_chains[i]) == 1:
         if polarities[i - 1] != polarities[i]:
            raise ValueError("polarity mismatch on single chain connect")
   for i, dg in enumerate(body_chains):
      chains[-1].append(dg[0])
      if i != 0:
         pol[len(chains) - 1] = polarities[i - 1]
      if len(dg) > 1:
         chains.extend([x] for x in dg[1:])
   for i, chain in enumerate(chains):
      if i in pol and pol[i] == "C":
         chains[i] = chains[i][::-1]
   return chains

def _dump_chainlist(cl, tag='cl', ich=0):
   for i, ap in enumerate(cl):
      import os
      fn = f'{tag}_iseg{ap.iseg}_chain{ich}_elem{i}.pdb'
      assert not os.path.exists(fn)
      ap.pose.dump_pdb(fn)

def _cyclic_permute_chains(chainslist, entrypol, exitpol):
   """rearrange segments in a cylic structure so chainbreak is at chain boundary
   """
   n2c = 'N' == entrypol[-1]

   # print('polarity', polarity)
   # for ich, ch in enumerate(chainslist):
   # _dump_chainlist(ch, 'before', ich)

   chainslist_beg = 0
   for i, cl in enumerate(chainslist):
      if any(x.cyclic_entry for x in cl):
         assert chainslist_beg == 0  # there can be only one
         chainslist_beg = i

   print('resplicing cyclic chains')
   beg, end = chainslist[chainslist_beg], chainslist[-1]

   # _dump_chainlist(ch, 'begend')

   if n2c:
      stub1 = util.get_bb_stubs(beg[0][0], [1])
      stub2 = util.get_bb_stubs(end[-1][0], [1])
      rm_lower_t(beg[0][0], 1)
      assert len(end[-1][0]) == 1
      end = end[:-1]
   else:
      stub1 = util.get_bb_stubs(beg[-1][0], [len(beg[-1][0])])
      stub2 = util.get_bb_stubs(end[0][0], [1])
      rm_upper_t(beg[-1][0], len(beg[-1][0]))
      assert len(end[0][0]) == 1
      end = end[1:]
   xalign = stub1[0] @ np.linalg.inv(stub2[0])

   all_one_chain = chainslist_beg + 1 == len(chainslist)  # beg is end, all one chain
   if all_one_chain:
      # no no, this doesn't work
      # chainslist[chainslist_beg] = end[-1:] + end[:-1] if n2c else end[1:] + end[:1]
      # chainslist.append([])  # caller expects empty end, have to add
      # raise NotImplementedE
      print("cyclic permute chains fail, probably forming a backbone cycle")
      raise ValueError('invalid cycle, involves backbone cycle with no chainbreaks')

   for p in end:
      util.xform_pose(xalign, p[0])
   chainslist[chainslist_beg] = end + beg if n2c else beg + end

   chainslist[-1] = []

def make_contorted_pose(
    entryexits,
    entrypol,
    exitpol,
    indices,
    from_seg,
    to_seg,
    origin_seg,
    seg_pos,
    position,
    is_cyclic,
    align,
    cryst_info,
    end,
    iend,
    only_connected,
    join,
    cyclic_permute,
    cyclictrim,
    provenance,
    make_chain_list,
    full_output_segs=[],
):  # yapf: disable
   """there be dragons here"""
   nseg = len(entryexits)
   entryexits, rest = zip(*entryexits)
   for ap in it.chain(*entryexits, *rest):
      assert isinstance(ap, AnnoPose)
   chainslist = reorder_spliced_as_N_to_C(entryexits, entrypol[1:iend])

   if align:
      for ap in it.chain(*chainslist, *rest):
         util.xform_pose(position, ap.pose)
   if cyclic_permute and len(chainslist) > 1:
      cyclic_entry_count = 0
      for ap in it.chain(*entryexits, *rest):
         cyclic_entry_count += ap.cyclic_entry is not None
      assert cyclic_entry_count == 1
      _cyclic_permute_chains(chainslist, entrypol, exitpol)
      assert len(chainslist[-1]) == 0
      chainslist = chainslist[:-1]
   sourcelist = [[x[1] for x in c] for c in chainslist]
   chainslist = [[x[0] for x in c] for c in chainslist]

   ret_chain_list = []
   pose = ros.core.pose.Pose()
   prov0 = []
   splicepoints = []
   for i, (chains, sources) in enumerate(zip(chainslist, sourcelist)):
      if (only_connected and len(chains) is 1 and (end or chains is not chainslist[-1])):
         skipsegs = (to_seg, from_seg) if not is_cyclic else []
         skipsegs = [nseg - 1 if x is -1 else x for x in skipsegs]
         skipsegs = [s for s in skipsegs if s not in full_output_segs]
         if origin_seg is not None:
            skipsegs.append(origin_seg)

         if (only_connected == "auto" and sources[0][0] in skipsegs) or only_connected != "auto":
            # print('skip', i, skipsegs, len(chains), len(sources))
            continue
      if make_chain_list:
         ret_chain_list.append(chains[0])
      ros.core.pose.append_pose_to_pose(pose, chains[0], True)
      prov0.append(sources[0])
      prev_source, prev_chain = sources[0], chains[0]
      for chain, source in zip(chains[1:], sources[1:]):
         assert isinstance(chain, ros.core.pose.Pose)
         rm_upper_t(pose, len(pose))
         rm_lower_t(chain, 1)
         splicepoints.append(len(pose))
         if make_chain_list:
            ret_chain_list.append(chain)
         fixres = len(pose)

         ####################################################

         # tim look here!
         # pose: overall output structure
         # chain: new thing to splice on, already truncated
         # prev_chain: the last thing spliced on, already truncated
         # pose_source_before: untruncated pose of previous BB
         # pose_source_after: untruncated pose of this BB
         # lb/ub_before/after residue range in source poses in truncated 'chain's

         ros.core.pose.append_pose_to_pose(pose, chain, not join)
         iseg_before, pose_source_before, lb_before, ub_before = prev_source
         iseg_after, pose_source_after, lb_after, ub_after = source

         #

         ######################################################

         # this dosen't work correctly
         # util.fix_bb_o(pose, fixres)
         # util.fix_bb_h(pose, fixres + 1)
         prov0.append(source)
         prev_source, prev_chain = source, chain
   if not only_connected or only_connected == "auto":
      for chain, source in it.chain(*rest):
         assert isinstance(chain, ros.core.pose.Pose)
         if make_chain_list:
            ret_chain_list.append(chain)
         ros.core.pose.append_pose_to_pose(pose, chain, True)
         prov0.append(source)
   assert util.worst_CN_connect(pose) < 0.5
   assert util.no_overlapping_adjacent_residues(pose)

   if cryst_info:
      ci = pyrosetta.rosetta.core.io.CrystInfo()
      ci.A(cryst_info[0])  # cell dimensions
      ci.B(cryst_info[1])
      ci.C(cryst_info[2])
      ci.alpha(cryst_info[3])  # cell angles
      ci.beta(cryst_info[4])
      ci.gamma(cryst_info[5])
      ci.spacegroup(cryst_info[6])  # sace group
      pi = pyrosetta.rosetta.core.pose.PDBInfo(pose)
      pi.set_crystinfo(ci)
      pose.pdb_info(pi)

   ros.core.scoring.dssp.Dssp(pose).insert_ss_into_pose(pose)

   if not provenance and make_chain_list:
      return pose, ret_chain_list
   if not provenance:
      return pose, splicepoints
   prov = []
   for i, pr in enumerate(prov0):
      iseg, psrc, lb0, ub0 = pr
      lb1 = sum(ub - lb + 1 for _, _, lb, ub in prov0[:i]) + 1
      ub1 = lb1 + ub0 - lb0
      if ub0 == lb0:
         assert cyclic_permute
         continue
      assert ub0 - lb0 == ub1 - lb1
      assert 0 < lb0 <= len(psrc) and 0 < ub0 <= len(psrc)
      assert 0 < lb1 <= len(pose) and 0 < ub1 <= len(pose)
      # if psrc.sequence()[lb0 - 1:ub0] != pose.sequence()[lb1 - 1:ub1]:
      # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      assert psrc.sequence()[lb0 - 1:ub0] == pose.sequence()[lb1 - 1:ub1]
      prov.append((lb1, ub1, psrc, lb0, ub0))
   if make_chain_list:
      return pose, prov, ret_chain_list
   return pose, prov
