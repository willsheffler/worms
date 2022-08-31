from deferred_import import deferred_import
from functools import lru_cache
from worms.pose_contortions import contort_pose_chains, make_contorted_pose
from collections.abc import Iterable
import numpy as np
import willutil as wu
import worms

ros = deferred_import('worms.rosetta_init')

def make_pose_crit(
   bbdb,
   ssdag,
   criteria,
   indices,
   positions,
   only_connected="auto",
   provenance=False,
   join=True,
   full_output_segs=[],
   alignto='middle',
   **kw,
):
   cryst_info = None
   if hasattr(criteria, "crystinfo"):  # TODO should this be here in general?
      cryst_info = criteria.crystinfo(segpos=positions)
      # assert cryst_info is not None

   return make_pose(
      bbdb=bbdb,
      ssdag=ssdag,
      indices=indices,
      positions=positions,
      only_connected=only_connected,
      provenance=provenance,
      join=join,
      from_seg=criteria.from_seg,
      to_seg=criteria.to_seg,
      origin_seg=criteria.origin_seg,
      is_cyclic=criteria.is_cyclic,
      position=criteria.alignment(positions, alignto=alignto),
      cryst_info=cryst_info,
      full_output_segs=full_output_segs,
      **kw,
   )

def make_pose(
      bbdb,
      ssdag,
      indices,
      positions,
      only_connected="auto",
      provenance=False,
      join=True,
      from_seg=0,
      to_seg=-1,
      origin_seg=None,
      is_cyclic=False,
      position=np.eye(4),
      cryst_info=None,
      full_output_segs=[],
      extensions=dict(),
      **kw,
):
   cyclic_info = [None] * 5
   if is_cyclic:
      cyclic_info[0] = from_seg, to_seg
      cyclic_info[1] = _dirn_to_polarity(ssdag.verts[-1].dirn[0])
      cyclic_info[2] = _dirn_to_polarity(ssdag.verts[0].dirn[1])
      cyclic_info[3] = ssdag.verts[to_seg].ires[indices[to_seg], 0] + 1
      cyclic_info[4] = _dirn_to_polarity(ssdag.verts[to_seg].dirn[0])
      assert cyclic_info[3] != 0

   poses = _get_bb_poses(bbdb, ssdag, indices, **kw)

   entry_exit_chains = list()
   for ivert in range(len(indices)):
      x = 0
      if ivert in extensions:
         x = extensions[ivert]

      entry_exit_chains.append(
         _make_pose_single(
            poses[ivert],
            ssdag.verts[ivert],
            indices[ivert],
            positions[ivert],
            nverts=len(indices),
            ivert=ivert,
            cyclic_info=cyclic_info,
            nres_extension=x,
         ))

   # for i, e in enumerate(entry_exit_chains):
   #     for j, f in enumerate(e):
   #         for k, p in enumerate(f):
   #             print('dump_pdb test_%i_%i_%i.pdb' % (i, j, k))
   #             p.pose.dump_pdb('test_%i_%i_%i.pdb' % (i, j, k))

   result = make_contorted_pose(
      entryexits=entry_exit_chains,
      entrypol=_dirn_to_polarity(v.dirn[0] for v in ssdag.verts),
      exitpol=_dirn_to_polarity(v.dirn[1] for v in ssdag.verts),
      indices=indices,
      from_seg=from_seg,
      to_seg=to_seg,
      origin_seg=origin_seg,
      seg_pos=positions,
      position=position,
      is_cyclic=is_cyclic,
      align=True,
      cryst_info=cryst_info,
      end=(not is_cyclic),
      iend=None,  # (-1 if is_cyclic else None),
      only_connected=only_connected,
      join=join,
      cyclic_permute=is_cyclic,
      cyclictrim=cyclic_info[0],
      provenance=provenance,
      make_chain_list=False,
      full_output_segs=full_output_segs,
   )

   if not provenance:
      return result[0]
   return result

def make_pose_simple(
   ssdag,
   idx,
   xpos,
   only_spliced_regions=True,
   is_cyclic=False,
   **kw,
):
   kw = wu.Bunch(kw)

   structinfo = ssdag.get_structure_info(idx, **kw)
   srcposes = _get_bb_poses(kw.database.bblockdb, ssdag, idx, **kw)
   # for i, p in enumerate(srcposes):
   # p.dump_pdb(f'make_pose_simple_scrposes{i}.pdb')
   # assert 0
   if is_cyclic:
      assert len(structinfo.regions[-1]) == 1
      structinfo.regions[0][-1], structinfo.regions[-1][0] = (structinfo.regions[-1][0],
                                                              structinfo.regions[0][-1])
   provenance = list()

   newpose = worms.rosetta_init.Pose()
   for ireg, region in enumerate(structinfo.regions):
      print('make_pose_simple', ireg)
      for r in region:
         print('  ', repr(r))

      if len(region) == 1:
         if not only_spliced_regions:
            iseg = region[0].iseg
            pose = srcposes[iseg].clone()
            lb, ub = region[0].reswindow
            worms.util.rosetta_utils.xform_pose(xpos[iseg], pose)
            ros.core.pose.append_subpose_to_pose(newpose, pose, lb + 1, ub, True)
            # p = ros.core.pose.Pose()
            # ros.core.pose.append_subpose_to_pose(p, pose, lb + 1, ub, False)
            # p.dump_pdb(f'make_pose_simple_ireg{ireg}_iseg{chain.iseg}_{ich}_single.pdb')

      else:
         for ich, chain in enumerate(region):
            lb, ub = chain.reswindow
            pose = srcposes[chain.iseg].clone()
            worms.util.rosetta_utils.xform_pose(xpos[chain.iseg], pose)

            # p = ros.core.pose.Pose()
            # ros.core.pose.append_subpose_to_pose(p, pose, lb + 1, ub, False)
            # p.dump_pdb(f'make_pose_simple_ireg{ireg}_iseg{chain.iseg}_{ich}_multi.pdb')

            prov = [newpose.size() + 1, None, srcposes[chain.iseg], lb + 1, ub]
            ros.core.pose.append_subpose_to_pose(newpose, pose, lb + 1, ub, False)
            prov[1] = newpose.size()
            provenance.append(prov)

   return newpose.clone(), provenance

def _get_bb_poses(
      bbdb,
      ssdag,
      indices,
      extensions=dict(),
      **kw,
):
   poses = list()
   for iseg, (bbs, vert, idx) in enumerate(zip(ssdag.bbs, ssdag.verts, indices)):

      bb = bbs[vert.ibblock[idx]]
      # pdbfile = bytes(bb.file)
      # pdbfile = worms.util.tobytes(bb.file)
      pdbfile = worms.util.tostr(bb.file)

      # if iseg == kw['repeat_add_to_segment']:
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # bbdb.pose(pdbfile, **kw).dump_pdb('foo_0.pdb')
      # for i in range(1, 4):
      #    print('TEST ADD REPEATS', i, pdbfile)
      #    f = pdbfile + '?nrepeats=%i' % i
      #    pose = bbdb.pose(f, **kw)
      #    pose.dump_pdb(f'foo_{i}.pdb')
      # assert 0

      if iseg in extensions:
         # pose = bbdb.pose(pdbfile, **kw)
         # pose.dump_pdb('test0.pdb')
         pdbfile = worms.util.add_props_to_url(pdbfile, nrepeats=extensions[iseg])
         # pose = bbdb.pose(pdbfile, **kw)
         # pose.dump_pdb('test1.pdb')
         # assert 0
      poses.append(bbdb.pose(pdbfile, **kw).clone())

      # if iseg in extensions:
      #    poses[-1].dump_pdb('foo_%i.pdb' % extensions[iseg])
      #    if extensions[iseg] == 3: assert 0

   return poses

@lru_cache(maxsize=1024)
def _get_pose_chains(pose):
   return list(pose.split_by_chain())

def _dirn_to_polarity(dirn):
   if isinstance(dirn, Iterable):
      return [_dirn_to_polarity(x) for x in dirn]
   return ["N", "C", "_"][dirn]

def _make_pose_single(
   pose,
   vert,
   idx,
   positions,
   nverts,
   ivert,
   cyclic_info,
   nres_extension=0,
):
   chains0 = _get_pose_chains(pose)
   start_of_chain = {i + 1: sum(len(c) for c in chains0[:i]) for i in range(len(chains0))}
   end_of_chain = {i + 1: sum(len(c) for c in chains0[:i + 1]) for i in range(len(chains0))}
   start_of_chain[None] = 0
   chains = {i + 1: c for i, c in enumerate(chains0)}

   ir_en = (vert.ires[idx, 0] + 1) or -1
   ir_ex = (vert.ires[idx, 1] + 1) or -1
   if ir_en < ir_ex:
      ir_ex += nres_extension
   else:
      ir_en += nres_extension

   return contort_pose_chains(
      pose=pose,
      chains=chains,
      nseg=nverts,
      ir_en=ir_en,
      ir_ex=ir_ex,
      pl_en=_dirn_to_polarity(vert.dirn[0]),
      pl_ex=_dirn_to_polarity(vert.dirn[1]),
      chain_start=start_of_chain,
      chain_end=end_of_chain,
      position=positions,
      pad=(0, 0),
      iseg=ivert,
      cyclictrim=cyclic_info[0],
      last_seg_entrypol=cyclic_info[1],
      first_seg_exitpol=cyclic_info[2],
      sym_ir=cyclic_info[3],
      sym_pol=cyclic_info[4],
   )
