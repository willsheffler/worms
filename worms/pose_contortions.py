"""TODO: Summary
"""

import os
import itertools as it
from collections import defaultdict, OrderedDict

from worms import util

try:
    import pyrosetta
    from pyrosetta import rosetta as ros
except ImportError:
    pass


class AnnoPose:
    """TODO: Summary

    Attributes:
        cyclic_entry (TYPE): Description
        iseg (TYPE): Description
        pose (TYPE): Description
        src_lb (TYPE): Description
        src_ub (TYPE): Description
        srcpose (TYPE): Description
    """

    def __init__(self, pose, iseg, srcpose, src_lb, src_ub, cyclic_entry):
        """TODO: Summary

        Args:
            pose (TYPE): Description
            iseg (TYPE): Description
            srcpose (TYPE): Description
            src_lb (TYPE): Description
            src_ub (TYPE): Description
            cyclic_entry (TYPE): Description
        """
        self.pose = pose
        self.iseg = iseg
        self.srcpose = srcpose
        self.src_lb = src_lb
        self.src_ub = src_ub
        self.cyclic_entry = cyclic_entry

    def __iter__(self):
        """TODO: Summary

        Yields:
            TYPE: Description
        """
        yield self.pose
        yield (self.iseg, self.srcpose, self.src_lb, self.src_ub)

    def __getitem__(self, i):
        """TODO: Summary

        Args:
            i (TYPE): Description
        """
        if i is 0: return self.pose
        if i is 1: return (self.iseg, self.srcpose, self.src_lb, self.src_ub)

    def seq(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return self.pose.sequence()

    def srcseq(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return self.srcpose.sequence()[self.src_lb - 1:self.src_ub]


def make_pose_chains_from_seg(
        segment,
        indices,
        position=None,
        pad=(0, 0),
        iseg=None,
        segments=None,
        cyclictrim=None,
):
    """what a monster this has become. returns (segchains, rest)
    segchains elems are [enterexitchain] or, [enterchain, ..., exitchain]
    rest holds other chains IFF enter and exit in same chain
    each element is a pair [pose, source] where source is
    (origin_pose, start_res, stop_res)
    cyclictrim specifies segments which are spliced across the
    symmetric interface. segments only needed if cyclictrim==True
    if cyclictrim, last segment will only be a single entry residue

    Args:
        indices (TYPE): Description
        position (None, optional): Description
        pad (tuple, optional): Description
        iseg (None, optional): Description
        segments (None, optional): Description
        cyclictrim (None, optional): Description

    Returns:
        TYPE: Description
    """

    if isinstance(indices, int):
        assert not cyclictrim
        index = indices
    else:
        index = indices[iseg]

    spliceable = segment.spliceables[segment.bodyid[index]]
    ir_en, ir_ex = segment.entryresid[index], segment.exitresid[index]
    pl_en, pl_ex = segment.entrypol, segment.exitpol
    pose, chains = spliceable.body, spliceable.chains
    chain_start = spliceable.start_of_chain
    chain_end = spliceable.end_of_chain
    nseg = len(segments) if segments else 0
    if segments and cyclictrim:
        last_seg_entrypol = segments[-1].entrypol
        first_seg_exitpol = segments[0].exitpol
        sym_ir = segments[cyclictrim[1]].entryresid[indices[cyclictrim[1]]]
        sym_pol = segments[cyclictrim[1]].entrypol
    else:
        last_seg_entrypol = first_seg_exitpol = sym_ir = sym_pol = None

    return contort_pose_chains(
        pose,
        chains,
        nseg,
        ir_en,
        ir_ex,
        pl_en,
        pl_ex,
        chain_start,
        chain_end,
        position=position,
        pad=pad,
        iseg=iseg,
        cyclictrim=cyclictrim,
        last_seg_entrypol=last_seg_entrypol,
        first_seg_exitpol=first_seg_exitpol,
        sym_ir=sym_ir,
        sym_pol=sym_pol,
    )


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
            assert sym_ch != entry_chain or sym_pol != entrypol
            assert sym_ch != exit_chain or sym_pol != exitpol
        # annotate enex entries with cyclictrim info
        cyclic_entry[pose.chain(sym_ir)] = (iseg, sym_ir, sym_pol)
    if cyclictrim and iseg == cyclictrim[1]:
        assert ir_ex == -1
        assert iseg + 1 == nseg
        i = ir_en
        p = util.subpose(pose, i, i)
        if position is not None: util.xform_pose(position, p)
        return [AnnoPose(p, iseg, pose, i, i, None)], []

    ch_en = pose.chain(ir_en) if ir_en > 0 else None
    ch_ex = pose.chain(ir_ex) if ir_ex > 0 else None

    if cyclictrim and iseg == 0:
        pl_en = last_seg_entrypol
    if cyclictrim and iseg + 1 == nseg:
        assert 0, 'last segment not returned as single entry residue?!?'
        pl_ex = first_seg_exitpol
    if ch_en: ir_en -= chain_start[ch_en]
    if ch_ex: ir_ex -= chain_start[ch_ex]
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
            print('cyclictrim_in_rest', cyclictrim_in_rest,
                  'did_cyclictrim_in_rest', did_cyclictrim_in_rest)
            print('iseg', iseg, 'len(chains)', len(chains))
            assert cyclictrim_in_rest == did_cyclictrim_in_rest
    if ch_en: del rest[chains[ch_en]]
    if ch_en == ch_ex:
        assert len(rest) + 1 == len(chains)
        p, l1, u1 = util.trim_pose(chains[ch_en], ir_en, pl_en, pad[0])
        iexit1 = ir_ex - (pl_ex == 'C') * (len(chains[ch_en]) - len(p))
        p, l2, u2 = util.trim_pose(p, iexit1, pl_ex, pad[1] - 1)
        lb = l1 + l2 - 1 + chain_start[ch_en]
        ub = l1 + u2 - 1 + chain_start[ch_en]
        enex = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_en])]
        assert p.sequence() == pose.sequence()[lb - 1:ub]
        rest = list(rest.values())
    else:
        if ch_ex: del rest[chains[ch_ex]]
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
                  'proceed with caution and tell will to fix!')
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
    return enex, rest
