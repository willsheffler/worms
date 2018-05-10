"""TODO: Summary
"""
import multiprocessing
import os
import itertools as it
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.linalg import inv
try:
    import pyrosetta
    from pyrosetta import rosetta as ros
    from pyrosetta.rosetta.core import scoring
except ImportError:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('pyrosetta not available, worms won\'t work')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
from worms import util
import inspect

from worms.pose_contortions import (make_contorted_pose, contort_pose_chains,
                                    AnnoPose)


class SpliceSite:
    """TODO: Summary

    Attributes:
        chain (TYPE): Description
        polarity (TYPE): Description
        selections (TYPE): Description
    """

    def __init__(self, sele, polarity, chain=None):
        """TODO: Summary

        Args:
            sele (TYPE): Description
            polarity (TYPE): Description
            chain (None, optional): Description
        """
        if isinstance(sele, str) or isinstance(sele, int):
            sele = [sele]
        self.selections = list(sele)
        assert polarity in ('N', 'C', None)
        self.polarity = polarity
        self.chain = chain

    def resid(self, id, pose):
        """TODO: Summary

        Args:
            id (TYPE): Description
            pose (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        resid = id if id >= 0 else len(pose) + 1 + id
        if not 0 < resid <= len(pose):
            raise ValueError('resid ' + str(resid) +
                             ' invalid for pose of size ' + str(len(pose)))
        return resid

    def _resids_impl(self, sele, spliceable):
        """TODO: Summary

        Args:
            sele (TYPE): Description
            spliceable (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        if isinstance(sele, int):
            if self.chain is None:
                return set([self.resid(sele, spliceable.body)])
            else:
                ir = self.resid(sele, spliceable.chains[self.chain])
                ir += spliceable.start_of_chain[self.chain]
                return set([ir])
        elif isinstance(sele, str):
            x = sele.split(',')
            s = x[-1].split(':')
            chain = int(x[0]) if len(x) == 2 else None
            pose = spliceable.chains[chain] if chain else spliceable.body
            start = self.resid(int(s[0] or 1), pose)
            stop = self.resid(int(s[1] or -1), pose)
            step = int(s[2]) if len(s) > 2 else 1
            # print(start, stop + 1, step)
            resids = set()
            for ir in range(start, stop + 1, step):
                assert 0 < ir <= len(pose)
                resids.add(spliceable.start_of_chain[chain] + ir)
            return resids
        elif sele is None:
            return set([None])
        else:
            raise ValueError('selection must be int, str, or None')

    def _resids(self, spliceabe):
        """TODO: Summary

        Args:
            spliceabe (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        resids = set()
        for sele in self.selections:
            try:
                resids |= self._resids_impl(sele, spliceabe)
            except ValueError as e:
                raise ValueError(
                    'Error with selection ' + str(sele) + ': ' + str(e))
        resids = sorted(resids)
        if not resids:
            raise ValueError('empty SpliceSite')
        return resids

    def __repr__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        c = '' if self.chain is None else ', chain=' + str(self.chain)
        return 'SpliceSite(' + str(self.selections) + \
            ', ' + self.polarity + c + ')'


class Spliceable:
    """TODO: Summary

    Attributes:
        allowed_pairs (TYPE): Description
        body (TYPE): Description
        bodyid (TYPE): Description
        chains (TYPE): Description
        end_of_chain (TYPE): Description
        min_seg_len (TYPE): Description
        nsite (TYPE): Description
        sites (TYPE): Description
        start_of_chain (TYPE): Description
    """

    def __init__(self,
                 body,
                 sites,
                 *,
                 bodyid=None,
                 min_seg_len=1,
                 allowed_pairs=None):
        """TODO: Summary

        Args:
            body (TYPE): Description
            sites (TYPE): Description
            bodyid (None, optional): Description
            min_seg_len (int, optional): Description
            allowed_pairs (None, optional): Description

        Raises:
            ValueError: Description
        """
        self.body = body
        chains = list(body.split_by_chain())
        self.start_of_chain = {
            i + 1: sum(len(c) for c in chains[:i])
            for i in range(len(chains))
        }
        self.end_of_chain = {
            i + 1: sum(len(c) for c in chains[:i + 1])
            for i in range(len(chains))
        }
        self.start_of_chain[None] = 0
        self.chains = {i + 1: c for i, c in enumerate(chains)}
        self.bodyid = bodyid
        if callable(sites):
            sites = sites(body)
        if isinstance(sites, SpliceSite):
            sites = [sites]
        self.sites = list(sites)
        for i, site in enumerate(self.sites):
            if isinstance(site, str):
                raise ValueError('site currently must be (sele, polarity)')
            if not isinstance(site, SpliceSite):
                if isinstance(site, dict):
                    self.sites[i] = SpliceSite(**site)
                else:
                    if not isinstance(site, Iterable):
                        self.sites[i] = (site, )
                    self.sites[i] = SpliceSite(*site)
        self.nsite = dict(N=0, C=0)
        for s in self.sites:
            self.nsite[s.polarity] += 1
        self.min_seg_len = min_seg_len
        self._resids_list = [site._resids(self) for site in self.sites]
        self._len_body = len(body)
        self._chains = np.array([body.chain(i + 1) for i in range(len(body))])
        self.allowed_pairs = allowed_pairs

    def resids(self, isite):
        """TODO: Summary

        Args:
            isite (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isite < 0: return [None]
        return self._resids_list[isite]

    def spliceable_positions(self):
        """selection of resids, and map 'global' index to selected index

        Returns:
            TYPE: Description
        """
        resid_subset = set()
        for i in range(len(self.sites)):
            resid_subset |= set(self._resids_list[i])
        resid_subset = np.array(list(resid_subset))
        # really? must be an easier way to 'invert' a mapping in numpy?
        N = self._len_body + 1
        val, idx = np.where(0 == (
            np.arange(N)[np.newaxis, :] - resid_subset[:, np.newaxis]))
        to_subset = np.array(N * [-1])
        to_subset[idx] = val
        assert np.all(to_subset[resid_subset] == np.arange(len(resid_subset)))
        return resid_subset, to_subset

    def is_compatible(self, isite, ires, jsite, jres):
        """TODO: Summary

        Args:
            isite (TYPE): Description
            ires (TYPE): Description
            jsite (TYPE): Description
            jres (TYPE): Description

        Returns:
            TYPE: Description
        """
        if ires < 0 or jres < 0: return True
        assert 0 < ires <= self._len_body and 0 < jres <= self._len_body
        ichain, jchain = self._chains[ires - 1], self._chains[jres - 1]
        if ichain == jchain:
            ipol = self.sites[isite].polarity
            jpol = self.sites[jsite].polarity
            if ipol == jpol: return False
            if ipol == 'N': seglen = jres - ires + 1
            else: seglen = ires - jres + 1
            if seglen < self.min_seg_len: return False
        return True

    def sitepair_allowed(self, isite, jsite):
        """TODO: Summary

        Args:
            isite (TYPE): Description
            jsite (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isite == jsite:
            return False
        if isite < 0 or jsite < 0:
            return True
        if (self.allowed_pairs is not None
                and (isite, jsite) not in self.allowed_pairs):
            return False
        return True

    def __repr__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        sites = str([(s._resids(self), s.polarity) for s in self.sites])
        if len(sites) > 30:
            sites = sites[:30] + '...'
        return ('Spliceable: body=(' + str(self._len_body) + ',' + str(
            self.body).splitlines()[0].split('/')[-1] + '), sites=' + sites)

    # def __getstate__(self):
    # pdbfname = self.body.pdb_info().name() if self.body else None
    # return (pdbfname, self.sites, self.bodyid, self.min_seg_len)

    # def __setstate__(self, state):
    # body = pyrosetta.pose_from_file(state[0]) if state[0] else None
    # self.__init__(body, state[1], bodyid=state[2], min_seg_len=state[3])


def lineno():
    """Returns the current line number in our program.

    Returns:
        TYPE: Description
    """
    return inspect.currentframe().f_back.f_lineno


class Segment:
    """TODO: Summary

    Attributes:
        bodyid (TYPE): Description
        entrypol (TYPE): Description
        entryresid (TYPE): Description
        entrysiteid (TYPE): Description
        exitpol (TYPE): Description
        exitresid (TYPE): Description
        exitsiteid (TYPE): Description
        expert (TYPE): Description
        max_sites (TYPE): Description
        min_sites (TYPE): Description
        nchains (TYPE): Description
        spliceables (TYPE): Description
        x2exit (TYPE): Description
        x2orgn (TYPE): Description
    """

    def __init__(self, spliceables, entry=None, exit=None, expert=False):
        """TODO: Summary

        Args:
            spliceables (TYPE): Description
            entry (None, optional): Description
            exit (None, optional): Description
            expert (bool, optional): Description

        Raises:
            ValueError: Description
        """
        if entry and len(entry) is 2:
            entry, exit = entry
            if entry == '_': entry = None
            if exit == '_': exit = None
        self.entrypol = entry
        self.exitpol = exit
        self.expert = expert
        if entry not in ('C', 'N', None):
            raise ValueError('bad entry: "%s" type %s' % (entry, type(entry)))
        if exit not in ('C', 'N', None):
            raise ValueError('bad exit: "%s" type %s' % (exit, type(exit)))
        self.min_sites = dict(C=9e9, N=9e9)
        self.max_sites = dict(C=0, N=0)
        if not spliceables:
            raise ValueError('spliceables must not be empty, spliceables =' +
                             str(spliceables))
        for s in spliceables:
            if not isinstance(s, Spliceable):
                raise ValueError('Segment only accepts list of Spliceable')
        self.spliceables = list(spliceables)
        self.nchains = len(spliceables[0].chains)
        for s in spliceables:
            if not expert and len(s.chains) is not self.nchains:
                raise ValueError('different number of chains for spliceables',
                                 ' in segment (pass expert=True to ignore)')
            self.nchains = max(self.nchains, len(s.chains))
        self.resid_subset, self.to_subset, self.stubs = [], [], []
        for ibody, spliceable in enumerate(self.spliceables):
            for p in 'NC':
                self.min_sites[p] = min(self.min_sites[p], spliceable.nsite[p])
                self.max_sites[p] = max(self.max_sites[p], spliceable.nsite[p])
            resid_subset, to_subset = spliceable.spliceable_positions()
            stubs, _ = util.get_bb_stubs(spliceable.body, resid_subset)
            self.resid_subset.append(resid_subset)
            self.to_subset.append(to_subset)
            self.stubs.append(stubs)
        self.init_segment_data()

    def make_head(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        assert not (self.entrypol is None or self.exitpol is None)
        return Segment(
            self.spliceables,
            entry=self.entrypol,
            exit=None,
            expert=self.expert)

    def make_tail(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        assert not (self.entrypol is None or self.exitpol is None)
        return Segment(
            self.spliceables,
            entry=None,
            exit=self.exitpol,
            expert=self.expert)

    def merge_idx_slow(self, head, head_idx, tail, tail_idx):
        """TODO: Summary

        Args:
            head (TYPE): Description
            head_idx (TYPE): Description
            tail (TYPE): Description
            tail_idx (TYPE): Description

        Returns:
            TYPE: return joint index, -1 if head/tail pairing is invalid
        """
        "TODO THIS IS TOTALLY INEFFICEENT"
        head_idx, tail_idx = map(np.asarray, [head_idx, tail_idx])
        # print('merge_idx', head_idx.shape, tail_idx.shape)
        assert not (self.entrypol is None or self.exitpol is None)
        assert head.exitpol is None and tail.entrypol is None
        assert head_idx.shape == tail_idx.shape
        assert head_idx.ndim == 1
        idx = np.zeros_like(head_idx) - 1

        for i in range(len(idx)):
            tmp = np.where(
                (self.bodyid == head.bodyid[head_idx[i]]) *
                (self.entryresid == head.entryresid[head_idx[i]]) *
                (self.entrysiteid == head.entrysiteid[head_idx[i]]) *
                (self.bodyid == tail.bodyid[tail_idx[i]]) *
                (self.exitresid == tail.exitresid[tail_idx[i]]) *
                (self.exitsiteid == tail.exitsiteid[tail_idx[i]]))[0]
            assert len(tmp) <= 1
            if len(tmp) is 1:
                idx[i] = tmp[0]
        return idx

    def merge_idx(self, head, head_idx, tail, tail_idx):
        """TODO: Summary

        Args:
            head (TYPE): Description
            head_idx (TYPE): Description
            tail (TYPE): Description
            tail_idx (TYPE): Description

        Returns:
            TYPE: Description
        """
        ok1 = (head.bodyid[head_idx] == tail.bodyid[tail_idx])
        ok2 = (head.entrysiteid[head_idx] != tail.exitsiteid[tail_idx])
        ok = np.logical_and(ok1, ok2)
        return self.merge_idx_slow(head, head_idx[ok], tail, tail_idx[ok]), ok

    def split_idx(self, idx, head, tail):
        """return indices for separate head and tail segments

        Args:
            idx (TYPE): Description
            head (TYPE): Description
            tail (TYPE): Description

        Returns:
            TYPE: Description
        """
        assert not (self.entrypol is None or self.exitpol is None)
        assert head.exitpol is None and tail.entrypol is None
        assert idx.ndim == 1
        head_idx = np.zeros_like(idx) - 1
        tail_idx = np.zeros_like(idx) - 1
        for i in range(len(idx)):
            head_tmp = np.where(
                (self.bodyid[idx[i]] == head.bodyid) *
                (self.entryresid[idx[i]] == head.entryresid) *
                (self.entrysiteid[idx[i]] == head.entrysiteid))[0]
            tail_tmp = np.where(
                (self.bodyid[idx[i]] == tail.bodyid) *
                (self.exitresid[idx[i]] == tail.exitresid) *
                (self.exitsiteid[idx[i]] == tail.exitsiteid))[0]
            assert len(head_tmp) <= 1 and len(tail_tmp) <= 1
            # print(i, head_tmp, tail_tmp)
            if len(head_tmp) == 1 and len(tail_tmp) == 1:
                head_idx[i] = head_tmp[0]
                tail_idx[i] = tail_tmp[0]
        return head_idx, tail_idx

    def __len__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.bodyid)

    def init_segment_data(self):
        """TODO: Summary

        Raises:
            ValueError: Description
        """
        # print('init_segment_data', len(self.spliceables))
        # each array has all in/out pairs
        self.x2exit, self.x2orgn, self.bodyid = [], [], []
        self.entryresid, self.exitresid = [], []
        self.entrysiteid, self.exitsiteid = [], []
        # this whole loop is pretty inefficient, but that probably
        # doesn't matter much given the cost subsequent operations (?)
        for ibody, spliceable in enumerate(self.spliceables):
            for p in 'NC':
                self.min_sites[p] = min(self.min_sites[p], spliceable.nsite[p])
                self.max_sites[p] = max(self.max_sites[p], spliceable.nsite[p])
            # resid_subset, to_subset = spliceable.spliceable_positions()
            resid_subset = self.resid_subset[ibody]
            to_subset = self.to_subset[ibody]
            bodyid = ibody if spliceable.bodyid is None else spliceable.bodyid
            # extract 'stubs' from body at selected positions
            # rif 'stubs' have 'extra' 'features'... the raw field is
            # just bog-standard homogeneous matrices
            # stubs = rcl.bbstubs(spliceable.body, resid_subset)['raw']
            # stubs = stubs.astype('f8')
            # stubs = util.get_bb_stubs(spliceable.body, resid_subset)
            stubs = self.stubs[ibody]
            if len(resid_subset) != stubs.shape[0]:
                raise ValueError("no funny residues supported")
            stubs_inv = inv(stubs)
            entry_sites = (list(enumerate(spliceable.sites)) if self.entrypol
                           else [(-1,
                                  SpliceSite(
                                      sele=[None], polarity=self.entrypol))])
            exit_sites = (list(enumerate(spliceable.sites))
                          if self.exitpol else [(-1,
                                                 SpliceSite(
                                                     sele=[None],
                                                     polarity=self.exitpol))])
            for isite, entry_site in entry_sites:
                if entry_site.polarity != self.entrypol:
                    continue
                for ires in spliceable.resids(isite):
                    istub_inv = (np.eye(4)
                                 if not ires else stubs_inv[to_subset[ires]])
                    ires = ires or -1
                    for jsite, exit_site in exit_sites:
                        if (not spliceable.sitepair_allowed(isite, jsite)
                                or exit_site.polarity != self.exitpol):
                            continue
                        # print('pass')
                        for jres in spliceable.resids(jsite):
                            jstub = (np.eye(4)
                                     if not jres else stubs[to_subset[jres]])
                            jres = jres or -1
                            if not spliceable.is_compatible(
                                    isite, ires, jsite, jres):
                                continue
                            self.x2exit.append(istub_inv @ jstub)
                            self.x2orgn.append(istub_inv)
                            self.entrysiteid.append(isite)
                            self.entryresid.append(ires)
                            self.exitsiteid.append(jsite)
                            self.exitresid.append(jres)
                            self.bodyid.append(bodyid)
        if len(self.x2exit) is 0:
            raise ValueError('no valid splices found')
        self.x2exit = np.stack(self.x2exit)
        self.x2orgn = np.stack(self.x2orgn)
        self.entrysiteid = np.stack(self.entrysiteid)
        self.entryresid = np.array(self.entryresid)
        self.exitsiteid = np.array(self.exitsiteid)
        self.exitresid = np.array(self.exitresid)
        self.bodyid = np.array(self.bodyid)

    def same_bodies_as(self, other):
        """TODO: Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        bodies1 = [s.body for s in self.spliceables]
        bodies2 = [s.body for s in other.spliceables]
        return bodies1 == bodies2

    def make_pose_chains(
            self,
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

        spliceable = self.spliceables[self.bodyid[index]]
        ir_en, ir_ex = self.entryresid[index], self.exitresid[index]
        pl_en, pl_ex = self.entrypol, self.exitpol
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
            pose=pose,
            chains=chains,
            nseg=nseg,
            ir_en=ir_en,
            ir_ex=ir_ex,
            pl_en=pl_en,
            pl_ex=pl_ex,
            chain_start=chain_start,
            chain_end=chain_end,
            position=position,
            pad=pad,
            iseg=iseg,
            cyclictrim=cyclictrim,
            last_seg_entrypol=last_seg_entrypol,
            first_seg_exitpol=first_seg_exitpol,
            sym_ir=sym_ir,
            sym_pol=sym_pol,
        )


class Segments:
    """light wrapper around list of Segments

    Attributes:
        segments (TYPE): Description
    """

    def __init__(self, segments):
        """TODO: Summary

        Args:
            segments (TYPE): Description
        """
        self.segments = segments

    def __getitem__(self, idx):
        """TODO: Summary

        Args:
            idx (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(idx, slice):
            return Segments(self.segments[idx])
        return self.segments[idx]

    def __setitem__(self, idx, val):
        """TODO: Summary

        Args:
            idx (TYPE): Description
            val (TYPE): Description
        """
        self.segments[idx] = val

    def __iter__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return iter(self.segments)

    def __len__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.segments)

    def index(self, val):
        """TODO: Summary

        Args:
            val (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.segments.index(val)

    def split_at(self, idx):
        """TODO: Summary

        Args:
            idx (TYPE): Description

        Returns:
            TYPE: Description
        """
        tail, head = self[:idx + 1], self[idx:]
        tail[-1] = tail[-1].make_head()
        head[0] = head[0].make_tail()
        return tail, head


class Worms:
    """TODO: Summary

    Attributes:
        criteria (TYPE): Description
        detail (TYPE): Description
        indices (TYPE): Description
        positions (TYPE): Description
        score0 (TYPE): Description
        score0sym (TYPE): Description
        scores (TYPE): Description
        segments (TYPE): Description
        splicepoint_cache (dict): Description
    """

    def __init__(self, segments, scores, indices, positions, criteria, detail):
        """TODO: Summary

        Args:
            segments (TYPE): Description
            scores (TYPE): Description
            indices (TYPE): Description
            positions (TYPE): Description
            criteria (TYPE): Description
            detail (TYPE): Description
        """
        self.segments = segments
        self.scores = scores
        self.indices = indices
        self.positions = positions
        self.criteria = criteria
        self.detail = detail
        self.score0 = scoring.ScoreFunctionFactory.create_score_function(
            'score0')
        self.score0sym = scoring.symmetry.symmetrize_scorefunction(self.score0)
        self.splicepoint_cache = {}

    def pose(
            self,
            which,
            *,
            align=True,
            end=None,
            only_connected='auto',
            join=True,
            cyclic_permute=None,
            cyclictrim=None,
            provenance=False,
            make_chain_list=False,
            **kw
    ):  # yapf: disable
        """makes a pose for the ith worm"""
        if isinstance(which, Iterable):
            return (self.pose(
                w,
                align=align,
                end=end,
                join=join,
                only_connected=only_connected,
                **kw) for w in which)

        is_cyclic = self.criteria.is_cyclic
        from_seg, to_seg = self.criteria.from_seg, self.criteria.to_seg
        origin_seg = self.criteria.origin_seg
        seg_pos = self.positions[which]
        indices = self.indices[which]
        position = self.criteria.alignment(seg_pos)

        if end is None and cyclic_permute is None:
            cyclic_permute, end = is_cyclic, True
        if end is None:
            end = not is_cyclic or cyclic_permute
        if only_connected is None:
            only_connected = not is_cyclic
        if cyclic_permute is None:
            cyclic_permute = not end
        elif cyclic_permute and not is_cyclic:
            raise ValueError('cyclic_permute should only be used for Cyclic')
        if cyclictrim is None:
            cyclictrim = cyclic_permute
        if cyclictrim:
            cyclictrim = from_seg, to_seg

        nseg = len(self.segments)
        entrypol = [s.entrypol for s in self.segments]
        exitpol = [s.exitpol for s in self.segments]

        end = end or (not is_cyclic or cyclic_permute)
        iend = None if end else -1
        entryexits = [
            seg.make_pose_chains(
                indices,
                seg_pos[iseg],
                iseg=iseg,
                segments=self.segments,
                cyclictrim=cyclictrim)
            for iseg, seg in enumerate(self.segments[:iend])
        ]
        stupid_variable_return_type = make_contorted_pose(
            entryexits=entryexits,
            entrypol=entrypol,
            exitpol=exitpol,
            indices=indices,
            from_seg=from_seg,
            to_seg=to_seg,
            origin_seg=origin_seg,
            seg_pos=seg_pos,
            position=position,
            is_cyclic=is_cyclic,
            align=align,
            end=end,
            iend=iend,
            only_connected=only_connected,
            join=join,
            cyclic_permute=cyclic_permute,
            cyclictrim=cyclictrim,
            provenance=provenance,
            make_chain_list=make_chain_list,
        )

        assert len(stupid_variable_return_type) == 2
        if provenance:
            return stupid_variable_return_type
        else:
            self.splicepoint_cache[which] = stupid_variable_return_type[-1]
            return stupid_variable_return_type[0]
        return

    def splicepoints(self, which):
        """TODO: Summary

        Args:
            which (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not which in self.splicepoint_cache:
            self.pose(which)
        assert isinstance(which, int)
        return self.splicepoint_cache[which]

    def clear_caches(self):
        """TODO: Summary
        """
        self.splicepoint_cache = {}

    def sympose(
            self,
            which,
            score=False,
            provenance=False,
            fullatom=False,
            *,
            parallel=False,
            asym_score_thresh=50
    ):  # yapf: disable
        """TODO: Summary

        Args:
            which (TYPE): Description
            score (bool, optional): Description
            provenance (bool, optional): Description
            fullatom (bool, optional): Description
            parallel (bool, optional): Description
            asym_score_thresh (int, optional): Description

        Returns:
            TYPE: Description

        Raises:
            IndexError: Description
        """
        if isinstance(which, Iterable):
            which = list(which)
            if not all(0 <= i < len(self) for i in which):
                raise IndexError('invalid worm index')
            if parallel:
                with ThreadPoolExecutor() as pool:
                    result = pool.map(self.sympose, which, it.repeat(score),
                                      it.repeat(provenance),
                                      it.repeat(fullatom))
                    return list(result)
            else:
                return list(
                    map(self.sympose, which, it.repeat(score),
                        it.repeat(provenance)))
        if not 0 <= which < len(self):
            raise IndexError('invalid worm index')
        p, prov = self.pose(which, provenance=True)
        if fullatom: pfull = p.clone()
        pcen = p
        # todo: why is asym scoring broken?!?
        # try: score0asym = self.score0(p)
        # except: score0asym = 9e9
        # if score0asym > asym_score_thresh:
        # return None, None if score else None
        ros.core.util.switch_to_residue_type_set(pcen, 'centroid')
        if self.criteria.symfile_modifiers:
            symdata = util.get_symdata_modified(
                self.criteria.symname,
                **self.criteria.symfile_modifiers(
                    segpos=self.positions[which]))
        else:
            symdata = util.get_symdata(self.criteria.symname)
        sfxn = self.score0sym
        if symdata is None:
            sfxn = self.score0
        else:
            ros.core.pose.symmetry.make_symmetric_pose(pcen, symdata)
        if fullatom:
            if symdata is not None:
                ros.core.pose.symmetry.make_symmetric_pose(pfull, symdata)
            p = pfull
        else:
            p = pcen
        if score and provenance:
            return p, sfxn(pcen), prov
        if score:
            return p, sfxn(pcen)
        if provenance:
            return p, prov
        return p

    def splices(self, which):
        """TODO: Summary

        Args:
            which (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(which, Iterable): return (self.splices(w) for w in which)
        splices = []
        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i]
            isegchoice1 = self.indices[which, i]
            ibody1 = seg1.bodyid[isegchoice1]
            spliceable1 = seg1.spliceables[ibody1]
            resid1 = seg1.exitresid[isegchoice1]
            ichain1 = spliceable1.body.chain(resid1)
            chainresid1 = resid1 - spliceable1.start_of_chain[ichain1]
            seg2 = self.segments[i + 1]
            isegchoice2 = self.indices[which, i + 1]
            ibody2 = seg2.bodyid[isegchoice2]
            spliceable2 = seg2.spliceables[ibody2]
            resid2 = seg2.entryresid[isegchoice2]
            ichain2 = spliceable2.body.chain(resid2)
            chainresid2 = resid2 - spliceable2.start_of_chain[ichain2]
            drn = self.segments[i].exitpol + self.segments[i + 1].entrypol
            splices.append((ibody1, ichain1, chainresid1, ibody2, ichain2,
                            chainresid2, drn))
        return splices

    def __len__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.scores)

    def __getitem__(self, i):
        """TODO: Summary

        Args:
            i (TYPE): Description

        Returns:
            TYPE: Description
        """
        return (
            i,
            self.scores[i],
        ) + self.sympose(
            i, score=True)
