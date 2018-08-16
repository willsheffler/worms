from functools import lru_cache
from worms.pose_contortions import contort_pose_chains, make_contorted_pose
from collections.abc import Iterable
import numpy as np


def make_pose_crit(
        bbdb,
        ssdag,
        criteria,
        indices,
        positions,
        only_connected='auto',
        provenance=False,
        join=True,
):
    cryst_info = None
    if hasattr(criteria, 'crystinfo'):
        cryst_info = criteria.crystinfo(segpos=positions)

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
        position=criteria.alignment(positions),
        cryst_info=cryst_info,
    )


def make_pose(
        bbdb,
        ssdag,
        indices,
        positions,
        only_connected='auto',
        provenance=False,
        join=True,
        from_seg=0,
        to_seg=-1,
        origin_seg=None,
        is_cyclic=False,
        position=np.eye(4),
        cryst_info=None,
):
    cyclic_info = [None] * 5
    if is_cyclic:
        cyclic_info[0] = from_seg, to_seg
        cyclic_info[1] = _dirn_to_polarity(ssdag.verts[-1].dirn[0])
        cyclic_info[2] = _dirn_to_polarity(ssdag.verts[0].dirn[1])
        cyclic_info[3] = ssdag.verts[to_seg].ires[indices[to_seg], 0] + 1
        cyclic_info[4] = _dirn_to_polarity(ssdag.verts[to_seg].dirn[0])
        assert cyclic_info[3] != 0

    poses = _get_bb_poses(bbdb, ssdag, indices)
    entry_exit_chains = list()
    for ivert in range(len(indices)):
        entry_exit_chains.append(
            _make_pose_single(
                poses[ivert],
                ssdag.verts[ivert],
                indices[ivert],
                positions[ivert],
                nverts=len(indices),
                ivert=ivert,
                cyclic_info=cyclic_info
            )
        )

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
        iend=None,  #(-1 if is_cyclic else None),
        only_connected=only_connected,
        join=join,
        cyclic_permute=is_cyclic,
        cyclictrim=cyclic_info[0],
        provenance=provenance,
        make_chain_list=False
    )

    if not provenance:
        return result[0]
    return result


def _get_bb_poses(bbdb, ssdag, indices):
    poses = list()
    for bbs, vert, idx in zip(ssdag.bbs, ssdag.verts, indices):
        bb = bbs[vert.ibblock[idx]]
        pdbfile = bytes(bb.file)
        poses.append(bbdb.pose(pdbfile))
    return poses


@lru_cache(maxsize=1024)
def _get_pose_chains(pose):
    return list(pose.split_by_chain())


def _dirn_to_polarity(dirn):
    if isinstance(dirn, Iterable):
        return [_dirn_to_polarity(x) for x in dirn]
    return ['N', 'C', '_'][dirn]


def _make_pose_single(pose, vert, idx, positions, nverts, ivert, cyclic_info):
    chains0 = _get_pose_chains(pose)
    start_of_chain = {
        i + 1: sum(len(c) for c in chains0[:i])
        for i in range(len(chains0))
    }
    end_of_chain = {
        i + 1: sum(len(c) for c in chains0[:i + 1])
        for i in range(len(chains0))
    }
    start_of_chain[None] = 0
    chains = {i + 1: c for i, c in enumerate(chains0)}
    return contort_pose_chains(
        pose=pose,
        chains=chains,
        nseg=nverts,
        ir_en=(vert.ires[idx, 0] + 1) or -1,
        ir_ex=(vert.ires[idx, 1] + 1) or -1,
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
