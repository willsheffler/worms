import multiprocessing
import os
import itertools as it
from collections.abc import Iterable
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from numpy.linalg import inv
from .criteria import CriteriaList
try:
    import pyrosetta
    from pyrosetta import rosetta as ros
    from pyrosetta.rosetta.core import scoring
    rm_lower_t = ros.core.pose.remove_lower_terminus_type_from_pose_residue
    rm_upper_t = ros.core.pose.remove_upper_terminus_type_from_pose_residue
except ImportError:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('pyrosetta not available, worms won\'t work')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
from . import util
import inspect


class SpliceSite:

    def __init__(self, sele, polarity, chain=None):
        if isinstance(sele, str) or isinstance(sele, int):
            sele = [sele]
        self.selections = list(sele)
        assert polarity in ('N', 'C', None)
        self.polarity = polarity
        self.chain = chain

    def resid(self, id, pose):
        resid = id if id >= 0 else len(pose) + 1 + id
        if not 0 < resid <= len(pose):
            raise ValueError('resid ' + str(resid)
                             + ' invalid for pose of size '
                             + str(len(pose)))
        return resid

    def _resids_impl(self, sele, spliceable):
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
        resids = set()
        for sele in self.selections:
            try:
                resids |= self._resids_impl(sele, spliceabe)
            except ValueError as e:
                raise ValueError('Error with selection '
                                 + str(sele) + ': ' + str(e))
        resids = sorted(resids)
        if not resids:
            raise ValueError('empty SpliceSite')
        return resids

    def __repr__(self):
        c = '' if self.chain is None else ', chain=' + str(self.chain)
        return 'SpliceSite(' + str(self.selections) + \
            ', ' + self.polarity + c + ')'


class Spliceable:

    def __init__(self, body, sites, *, bodyid=None, min_seg_len=1):
        self.body = body
        chains = list(body.split_by_chain())
        self.start_of_chain = {i + 1: sum(len(c) for c in chains[:i])
                               for i in range(len(chains))}
        self.end_of_chain = {i + 1: sum(len(c) for c in chains[:i + 1])
                             for i in range(len(chains))}
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
                        self.sites[i] = (site,)
                    self.sites[i] = SpliceSite(*site)
        self.nsite = dict(N=0, C=0)
        for s in self.sites: self.nsite[s.polarity] += 1
        self.min_seg_len = min_seg_len
        self._resids_list = [site._resids(self) for site in self.sites]
        self._len_body = len(body)
        self._chains = np.array([body.chain(i + 1) for i in range(len(body))])

    def resids(self, isite):
        if isite < 0: return [None]
        return self._resids_list[isite]

    def spliceable_positions(self):
        """selection of resids, and map 'global' index to selected index"""
        resid_subset = set()
        for i in range(len(self.sites)):
            resid_subset |= set(self._resids_list[i])
        resid_subset = np.array(list(resid_subset))
        # really? must be an easier way to 'invert' a mapping in numpy?
        N = self._len_body + 1
        val, idx = np.where(0 == (np.arange(N)[np.newaxis, :] -
                                  resid_subset[:, np.newaxis]))
        to_subset = np.array(N * [-1])
        to_subset[idx] = val
        assert np.all(to_subset[resid_subset] == np.arange(len(resid_subset)))
        return resid_subset, to_subset

    def is_compatible(self, isite, ires, jsite, jres):
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

    def __repr__(self):
        return ('Spliceable: body=(' + str(self._len_body) + ',' +
                str(self.body).splitlines()[0].split('/')[-1] +
                '), sites=' + str([(s._resids(self), s.polarity) for s in self.sites]))

    # def __getstate__(self):
        # pdbfname = self.body.pdb_info().name() if self.body else None
        # return (pdbfname, self.sites,
        # self.bodyid, self.min_seg_len)

    # def __setstate__(self, state):
        # body = pyrosetta.pose_from_file(state[0]) if state[0] else None
        # self.__init__(body, state[1], bodyid=state[2], min_seg_len=state[3])


class AnnoPose:

    def __init__(self, pose, iseg, srcpose, src_lb, src_ub, cyclic_entry):
        self.pose = pose
        self.iseg = iseg
        self.srcpose = srcpose
        self.src_lb = src_lb
        self.src_ub = src_ub
        self.cyclic_entry = cyclic_entry

    def __iter__(self):
        yield self.pose
        yield (self.iseg, self.srcpose, self.src_lb, self.src_ub)

    def __getitem__(self, i):
        if i is 0: return self.pose
        if i is 1: return (self.iseg, self.srcpose, self.src_lb, self.src_ub)

    def seq(self):
        return self.pose.sequence()

    def srcseq(self):
        return self.srcpose.sequence()[self.src_lb - 1:self.src_ub]


def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


class Segment:

    def __init__(self, spliceables, entry=None, exit=None, expert=False):
        if entry and len(entry) is 2:
            entry, exit = entry
            if entry == '_': entry = None
            if exit == '_': exit = None
        self.entrypol = entry or None
        self.exitpol = exit or None
        self.min_sites = dict(C=9e9, N=9e9)
        self.max_sites = dict(C=0, N=0)
        if not spliceables:
            raise ValueError('spliceables must not be empty, spliceables ='
                             + str(spliceables))
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
            stubs = util.get_bb_stubs(spliceable.body, resid_subset)
            self.resid_subset.append(resid_subset)
            self.to_subset.append(to_subset)
            self.stubs.append(stubs)
        self.init_segment_data()

    def __getstate__(self):
        state = dict(self.__dict__)
        # remove the big stuff
        if 'x2exit' in state: del state['x2exit']
        if 'x2orgn' in state: del state['x2orgn']
        if 'entrysiteid' in state: del state['entrysiteid']
        if 'entryresid' in state: del state['entryresid']
        if 'exitsiteid' in state: del state['exitsiteid']
        if 'exitresid' in state: del state['exitresid']
        if 'bodyid' in state: del state['bodyid']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_segment_data()  # recompute the big stuff

    def __len__(self):
        return len(self.bodyid)

    def init_segment_data(self):
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
            entry_sites = (list(enumerate(spliceable.sites)) if self.entrypol else
                           [(-1, SpliceSite(sele=[None], polarity=self.entrypol))])
            exit_sites = (list(enumerate(spliceable.sites)) if self.exitpol else
                          [(-1, SpliceSite(sele=[None], polarity=self.exitpol))])
            for isite, entry_site in entry_sites:
                if entry_site.polarity == self.entrypol:
                    for jsite, exit_site in exit_sites:
                        if isite != jsite and exit_site.polarity == self.exitpol:
                            for ires in spliceable.resids(isite):
                                istub_inv = (np.eye(4) if not ires
                                             else stubs_inv[to_subset[ires]])
                                ires = ires or -1
                                for jres in spliceable.resids(jsite):
                                    jstub = (np.eye(4) if not jres
                                             else stubs[to_subset[jres]])
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
        bodies1 = [s.body for s in self.spliceables]
        bodies2 = [s.body for s in other.spliceables]
        return bodies1 == bodies2

    def make_pose_chains(self, indices, position=None, pad=(0, 0), iseg=None,
                         segments=None, cyclictrim=None):
        """what a monster this has become. returns (segchains, rest)
        segchains elems are [enterexitchain] or, [enterchain, ..., exitchain]
        rest holds other chains IFF enter and exit in same chain
        each element is a pair [pose, source] where source is
        (origin_pose, start_res, stop_res)
        cyclictrim specifies segments which are spliced across the
        symmetric interface. segments only needed if cyclictrim==True
        if cyclictrim, last segment will only be a single entry residue
        """
        if isinstance(indices, int):
            assert not cyclictrim
            index = indices
        else: index = indices[iseg]
        spliceable = self.spliceables[self.bodyid[index]]
        pose, chains = spliceable.body, spliceable.chains
        ir_en, ir_ex = self.entryresid[index], self.exitresid[index]
        cyclic_entry = defaultdict(lambda: None)
        if cyclictrim and cyclictrim[1] < 0:
            cyclictrim = cyclictrim[0], cyclictrim[1] + len(segments)

        if cyclictrim and iseg == cyclictrim[0]:
            # assert ir_en == -1, 'paece sign not implemented yet'
            sym_ir = segments[cyclictrim[1]].entryresid[indices[cyclictrim[1]]]
            if ir_en == -1:
                ir_en = sym_ir
                cyclictrim_in_rest = False
            else:
                cyclictrim_in_rest = True
            # annotate enex entries with cyclictrim info
            cyclic_entry[pose.chain(sym_ir)] = (iseg, sym_ir,
                                                segments[-1].entrypol)
        if cyclictrim and iseg == cyclictrim[1]:
            assert ir_ex == -1
            assert iseg + 1 == len(segments)
            i = ir_en
            p = util.subpose(pose, i, i)
            if position is not None: util.xform_pose(position, p)
            return [AnnoPose(p, iseg, pose, i, i, None)], []
        ch_en = pose.chain(ir_en) if ir_en > 0 else None
        ch_ex = pose.chain(ir_ex) if ir_ex > 0 else None
        pl_en, pl_ex = self.entrypol, self.exitpol
        if cyclictrim and iseg == 0:
            pl_en = segments[-1].entrypol
        if cyclictrim and iseg + 1 == len(segments):
            assert 0
            pl_ex = segments[0].exitpol
        if ch_en: ir_en -= spliceable.start_of_chain[ch_en]
        if ch_ex: ir_ex -= spliceable.start_of_chain[ch_ex]
        assert ch_en or ch_ex
        rest = OrderedDict()
        did_cyclictrim_in_rest = False
        for i in range(1, len(chains) + 1):
            pchain = chains[i]
            lb = spliceable.start_of_chain[i] + 1
            ub = spliceable.end_of_chain[i]
            if cyclic_entry[i] is not None:
                if i not in (ch_en, ch_ex):
                    did_cyclictrim_in_rest = True
                ir = cyclic_entry[i][1] - spliceable.start_of_chain[i]
                pchain, lb, ub = util.trim_pose(pchain, ir, cyclic_entry[i][2])
                lb += spliceable.start_of_chain[i]
                ub += spliceable.start_of_chain[i]
            rest[chains[i]] = AnnoPose(pchain, iseg, pose, lb, ub,
                                       cyclic_entry[i])
            assert rest[chains[i]].seq() == rest[chains[i]].srcseq()
        if cyclictrim and iseg == cyclictrim[0]:
            assert cyclictrim_in_rest == did_cyclictrim_in_rest
        if ch_en: del rest[chains[ch_en]]
        if ch_en == ch_ex:
            assert len(rest) + 1 == len(chains)
            p, l1, u1 = util.trim_pose(chains[ch_en], ir_en, pl_en, pad[0])
            iexit1 = ir_ex - (pl_ex == 'C') * (len(chains[ch_en]) - len(p))
            p, l2, u2 = util.trim_pose(p, iexit1, pl_ex, pad[1] - 1)
            lb = l1 + l2 - 1 + spliceable.start_of_chain[ch_en]
            ub = l1 + u2 - 1 + spliceable.start_of_chain[ch_en]
            enex = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_en])]
            assert p.sequence() == pose.sequence()[lb - 1:ub]
            rest = list(rest.values())
        else:
            if ch_ex: del rest[chains[ch_ex]]
            p_en = [chains[ch_en]] if ch_en else []
            p_ex = [chains[ch_ex]] if ch_ex else []
            if p_en:
                p, lben, uben = util.trim_pose(p_en[0], ir_en, pl_en, pad[0])
                lb = lben + spliceable.start_of_chain[ch_en]
                ub = uben + spliceable.start_of_chain[ch_en]
                p_en = [AnnoPose(p, iseg, pose, lb, ub, cyclic_entry[ch_en])]
                assert p.sequence() == pose.sequence()[lb - 1:ub]
            if p_ex:
                p, lbex, ubex = util.trim_pose(p_ex[0], ir_ex, pl_ex,
                                               pad[1] - 1)
                lb = lbex + spliceable.start_of_chain[ch_ex]
                ub = ubex + spliceable.start_of_chain[ch_ex]
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
            for x in enex: x.pose = x.pose.clone()
            for x in rest: x.pose = x.pose.clone()
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


def _cyclic_permute_chains(chainslist, polarity):
    chainslist_beg = 0
    for i, cl in enumerate(chainslist):
        if any(x.cyclic_entry for x in cl):
            assert chainslist_beg == 0
            chainslist_beg = i
    beg, end = chainslist[chainslist_beg], chainslist[-1]
    # if chainslist_beg != 0:
    # raise NotImplementedError('peace sign not working yet')
    n2c = (polarity == 'N')
    if n2c:
        stub1 = util.get_bb_stubs(beg[0][0], [1])
        stub2 = util.get_bb_stubs(end[-1][0], [1])
        rm_lower_t(beg[0][0], 1)
        end = end[:-1]
    else:
        # from . import vis
        # for i, b in enumerate(beg): vis.showme(b[0], name='beg_%i' % i)
        # for i, e in enumerate(end): vis.showme(e[0], name='end_%i' % i)
        stub1 = util.get_bb_stubs(beg[-1][0], [len(beg[-1][0])])
        stub2 = util.get_bb_stubs(end[0][0], [1])
        rm_upper_t(beg[-1][0], len(beg[-1][0]))
        assert len(end[0][0]) == 1
        end = end[1:]
    xalign = stub1[0] @ np.linalg.inv(stub2[0])
    # print(xalign.shape)
    for p in end: util.xform_pose(xalign, p[0])
    if n2c: chainslist[chainslist_beg] = end + beg
    else: chainslist[chainslist_beg] = beg + end
    chainslist[-1] = []


def reorder_spliced_as_N_to_C(body_chains, polarities):
    "remap chains of each body such that concatenated chains are N->C"
    if len(body_chains) != len(polarities) + 1:
        raise ValueError('must be one more body_chains than polarities')
    chains, pol = [[]], {}
    if not all(0 < len(dg) for dg in body_chains):
        raise ValueError('body_chains values must be [enterexit], '
                         '[enter,exit], or [enter, ..., exit')
    for i in range(1, len(polarities)):
        if len(body_chains[i]) == 1:
            if polarities[i - 1] != polarities[i]:
                raise ValueError('polarity mismatch on single chain connect')
    for i, dg in enumerate(body_chains):
        chains[-1].append(dg[0])
        if i != 0: pol[len(chains) - 1] = polarities[i - 1]
        if len(dg) > 1: chains.extend([x] for x in dg[1:])
    for i, chain in enumerate(chains):
        if i in pol and pol[i] == 'C':
            chains[i] = chains[i][::-1]
    return chains


class Worms:

    def __init__(self, segments, scores, indices, positions, criteria, detail):
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

    def pose(self, which, *, align=True, end=None, only_connected='auto',
             join=True, cyclic_permute=None, cyclictrim=None,
             provenance=False, make_chain_list=False, **kw):
        "makes a pose for the ith worm"
        if isinstance(which, Iterable): return (
            self.pose(w, align=align, end=end, join=join,
                      only_connected=only_connected, **kw)
            for w in which)
        # print("Will needs to fix bb O/H position!")
        rm_lower_t = ros.core.pose.remove_lower_terminus_type_from_pose_residue
        rm_upper_t = ros.core.pose.remove_upper_terminus_type_from_pose_residue
        if end is None and cyclic_permute is None:
            cyclic_permute, end = self.criteria.is_cyclic, True
        if end is None:
            end = not self.criteria.is_cyclic or cyclic_permute
        if only_connected is None:
            only_connected = not self.criteria.is_cyclic
        if cyclic_permute is None:
            cyclic_permute = not end
        elif cyclic_permute and not self.criteria.is_cyclic:
            raise ValueError('cyclic_permute should only be used for Cyclic')
        if cyclictrim is None:
            cyclictrim = cyclic_permute
        if cyclictrim:
            cyclictrim = self.criteria.from_seg, self.criteria.to_seg
        iend = None if end else -1
        entryexits = [seg.make_pose_chains(self.indices[which],
                                           self.positions[which][iseg],
                                           iseg=iseg, segments=self.segments,
                                           cyclictrim=cyclictrim)
                      for iseg, seg in enumerate(self.segments[:iend])]
        entryexits, rest = zip(*entryexits)
        for ap in it.chain(*entryexits, *rest):
            assert isinstance(ap, AnnoPose)
        chainslist = reorder_spliced_as_N_to_C(
            entryexits, [s.entrypol for s in self.segments[1:iend]])
        if align:
            x = self.criteria.alignment(segpos=self.positions[which], **kw)
            for ap in it.chain(*chainslist, *rest): util.xform_pose(x, ap.pose)
        if cyclic_permute and len(chainslist) > 1:
            cyclic_entry_count = 0
            for ap in it.chain(*entryexits, *rest):
                cyclic_entry_count += (ap.cyclic_entry is not None)
            assert cyclic_entry_count == 1
            _cyclic_permute_chains(chainslist, self.segments[-1].entrypol)
            assert len(chainslist[-1]) == 0
            chainslist = chainslist[:-1]
        sourcelist = [[x[1] for x in c] for c in chainslist]
        chainslist = [[x[0] for x in c] for c in chainslist]
        ret_chain_list = []
        pose = ros.core.pose.Pose()
        prov0 = []
        splicepoints = []
        for chains, sources in zip(chainslist, sourcelist):
            if (only_connected and len(chains) is 1 and
                    (end or chains is not chainslist[-1])):
                skipsegs = ((self.criteria.to_seg, self.criteria.from_seg)
                            if not self.criteria.is_cyclic else [])
                skipsegs = [len(self.segments) - 1 if x is -1 else x
                            for x in skipsegs]
                if self.criteria.origin_seg is not None:
                    skipsegs.append(self.criteria.origin_seg)
                if ((only_connected == 'auto' and sources[0][0] in skipsegs)
                    or only_connected != 'auto'): continue
            if make_chain_list: ret_chain_list.append(chains[0])
            ros.core.pose.append_pose_to_pose(pose, chains[0], True)
            prov0.append(sources[0])
            for chain, source in zip(chains[1:], sources[1:]):
                assert isinstance(chain, ros.core.pose.Pose)
                rm_upper_t(pose, len(pose))
                rm_lower_t(chain, 1)
                splicepoints.append(len(pose))
                if make_chain_list: ret_chain_list.append(chain)
                ros.core.pose.append_pose_to_pose(pose, chain, not join)
                prov0.append(source)
        self.splicepoint_cache[which] = splicepoints
        if not only_connected or only_connected == 'auto':
            for chain, source in it.chain(*rest):
                assert isinstance(chain, ros.core.pose.Pose)
                if make_chain_list: ret_chain_list.append(chain)
                ros.core.pose.append_pose_to_pose(pose, chain, True)
                prov0.append(source)
        assert util.worst_CN_connect(pose) < 0.5
        assert util.no_overlapping_adjacent_residues(pose)
        if not provenance and make_chain_list: return pose, ret_chain_list
        if not provenance: return pose
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

    def splicepoints(self, which):
        if not which in self.splicepoint_cache:
            self.pose(which)
        assert isinstance(which, int)
        return self.splicepoint_cache[which]

    def clear_caches(self):
        self.splicepoint_cache = {}

    def sympose(self, which, score=False, provenance=False, fullatom=False, *,
                parallel=False, asym_score_thresh=50):
        if isinstance(which, Iterable):
            which = list(which)
            if not all(0 <= i < len(self) for i in which):
                raise IndexError('invalid worm index')
            if parallel:
                with ThreadPoolExecutor() as pool:
                    result = pool.map(self.sympose, which,
                                      it.repeat(score),
                                      it.repeat(provenance),
                                      it.repeat(fullatom))
                    return list(result)
            else: return list(map(self.sympose, which, it.repeat(score), it.repeat(provenance)))
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
            splices.append((ibody1, ichain1, chainresid1,
                            ibody2, ichain2, chainresid2, drn))
        return splices

    def __len__(self): return len(self.scores)

    def __getitem__(self, i):
        return (i, self.scores[i],) + self.sympose(i, score=True)


def _chain_xforms(segments):
    x2exit = [s.x2exit for s in segments]
    x2orgn = [s.x2orgn for s in segments]
    fullaxes = (np.newaxis,) * (len(x2exit) - 1)
    xconn = [x2exit[0][fullaxes], ]
    xbody = [x2orgn[0][fullaxes], ]
    for iseg in range(1, len(x2exit)):
        fullaxes = (slice(None),) + (np.newaxis,) * iseg
        xconn.append(xconn[iseg - 1] @ x2exit[iseg][fullaxes])
        xbody.append(xconn[iseg - 1] @ x2orgn[iseg][fullaxes])
    perm = list(range(len(xbody) - 1, -1, -1)) + [len(xbody), len(xbody) + 1]
    xbody = [np.transpose(x, perm) for x in xbody]
    xconn = [np.transpose(x, perm) for x in xconn]
    return xbody, xconn


__print_best = False
__best_score = 9e9


def _grow_chunk(samp, segpos, conpos, context):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    _, _, segs, end, criteria, thresh, matchlast, _, max_results = context
    # print('_grow_chunk', samp, end, thresh, matchlast, max_results)
    ML = matchlast
    # body must match, and splice sites must be distinct
    if ML is not None:
        # print('  ML')
        ndimchunk = segpos[0].ndim - 2
        bidB = segs[-1].bodyid[samp[-1]]
        site3 = segs[-1].entrysiteid[samp[-1]]
        if ML < ndimchunk:
            bidA = segs[ML].bodyid
            site1 = segs[ML].entrysiteid
            site2 = segs[ML].exitsiteid
            allowed = (bidA == bidB) * (site1 != site3) * (site2 != site3)
            idx = (slice(None),) * ML + (allowed,)
            segpos = segpos[: ML] + [x[idx] for x in segpos[ML:]]
            conpos = conpos[: ML] + [x[idx] for x in conpos[ML:]]
            idxmap = np.where(allowed)[0]
        else:
            bidA = segs[ML].bodyid[samp[ML - ndimchunk]]
            site1 = segs[ML].entrysiteid[samp[ML - ndimchunk]]
            site2 = segs[ML].exitsiteid[samp[ML - ndimchunk]]
            if bidA != bidB or site3 == site2 or site3 == site1:
                return
    segpos, conpos = segpos[:end], conpos[:end]
    # print('  do geom')
    for iseg, seg in enumerate(segs[end:]):
        segpos.append(conpos[-1] @ seg.x2orgn[samp[iseg]])
        if seg is not segs[-1]:
            conpos.append(conpos[-1] @ seg.x2exit[samp[iseg]])
    # print('  scoring')
    score = criteria.score(segpos=segpos)
    # print('  scores shape', score.shape)
    if __print_best:
        global __best_score
        min_score = np.min(score)
        if min_score < __best_score:
            __best_score = min_score
            if __best_score < thresh * 5:
                print('best for pid %6i %7.3f' % (os.getpid(), __best_score))
    # print('  trimming max_results')
    ilow0 = np.where(score < thresh)
    if len(ilow0) > max_results:
        order = np.argsort(score[ilow0])
        ilow0 = ilow0[order[:max_results]]
    sampidx = tuple(np.repeat(i, len(ilow0[0])) for i in samp)
    lowpostmp = []
    # print('  make lowpos')
    for iseg in range(len(segpos)):
        ilow = ilow0[: iseg + 1] + (0,) * (segpos[0].ndim - 2 - (iseg + 1))
        lowpostmp.append(segpos[iseg][ilow])
    ilow1 = (ilow0 if (ML is None or ML >= ndimchunk) else
             ilow0[:ML] + (idxmap[ilow0[ML]],) + ilow0[ML + 1:])
    return score[ilow0], np.array(ilow1 + sampidx).T, np.stack(lowpostmp, 1)


def _grow_chunks(ijob, context):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    sampsizes, njob, segments, end, _, _, _, every_other, max_results = context
    samples = list(util.MultiRange(sampsizes)[ijob::njob * every_other])
    segpos, connpos = _chain_xforms(segments[:end])  # common data
    args = [samples, it.repeat(segpos),
            it.repeat(connpos), it.repeat(context)]
    chunks = list(map(_grow_chunk, *args))
    chunks = [c for c in chunks if c is not None]
    if not chunks: return None
    scores = np.concatenate([x[0] for x in chunks])
    lowidx = np.concatenate([x[1] for x in chunks])
    lowpos = np.concatenate([x[2] for x in chunks])
    order = np.argsort(scores)
    return [scores[order[:max_results]],
            lowidx[order[:max_results]],
            lowpos[order[:max_results]]]


def _check_topology(segments, criteria, expert=False):
    if segments[0].entrypol is not None:
        raise ValueError('beginning of worm cant have entry')
    if segments[-1].exitpol is not None:
        raise ValueError('end of worm cant have exit')
    for a, b in zip(segments[:-1], segments[1:]):
        if not (a.exitpol and b.entrypol and a.exitpol != b.entrypol):
            raise ValueError('incompatible exit->entry polarity: '
                             + str(a.exitpol) + '->'
                             + str(b.entrypol) + ' on segment pair: '
                             + str((segments.index(a), segments.index(b))))
    matchlast = criteria.last_body_same_as
    if matchlast is not None and not expert and (
            not segments[matchlast].same_bodies_as(segments[-1])):
        raise ValueError("segments[matchlast] not same as segments[-1], "
                         + "if you're sure, pass expert=True")
    if criteria.is_cyclic and not criteria.to_seg in (-1, len(segments) - 1):
        raise ValueError('Cyclic and to_seg is not last segment,'
                         'if you\'re sure, pass expert=True')
    if criteria.is_cyclic:
        beg, end = segments[criteria.from_seg], segments[criteria.to_seg]
        sites_required = {'N': 0, 'C': 0, None: 0}
        sites_required[beg.entrypol] += 1
        sites_required[beg.exitpol] += 1
        sites_required[end.entrypol] += 1
        # print('pols', beg.entrypol, beg.exitpol, end.entrypol)
        for pol in 'NC':
            # print(pol, beg.max_sites[pol], sites_required[pol])
            if beg.max_sites[pol] < sites_required[pol]:
                msg = 'Not enough %s sites in any of segment %i Spliceables, %i required, at most %i available' % (
                    pol, criteria.from_seg, sites_required[pol],
                    beg.max_sites[pol])
                raise ValueError(msg)
            if beg.min_sites[pol] < sites_required[pol]:
                msg = 'Not enough %s sites in all of segment %i Spliceables, %i required, some have only %i available (pass expert=True if you really want to run anyway)' % (
                    pol, criteria.from_seg, sites_required[pol],
                    beg.max_sites[pol])
                if not expert: raise ValueError(msg)
                print("WARNING:", msg)
    return matchlast


def grow(
    segments,
    criteria,
    *,
    thresh=2,
    expert=0,
    memsize=1e6,
    executor=None,
    executor_args=None,
    max_workers=None,
    verbosity=0,
    chunklim=None,
    max_samples=int(1e12),
    max_results=int(1e4)
):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    if isinstance(executor, (ProcessPoolExecutor, ThreadPoolExecutor)):
        raise ValueError('please use dask.distributed executor')
    if verbosity > 0:
        print('grow, from', criteria.from_seg, 'to', criteria.to_seg)
        for i, seg in enumerate(segments):
            print(' segment', i, 'enter:', seg.entrypol, 'exit:', seg.exitpol)
            for sp in seg.spliceables: print('   ', sp)
    if verbosity > 1:
        global __print_best
        __print_best = True
    if not isinstance(criteria, CriteriaList):
        criteria = CriteriaList(criteria)
    if max_workers is not None and max_workers <= 0:
        max_workers = util.cpu_count()
    if executor_args is None and max_workers is None:
        executor_args = dict()
    elif executor_args is None:
        executor_args = dict(max_workers=max_workers)
    elif executor_args is not None and max_workers is not None:
        raise ValueError('executor_args incompatible with max_workers')

    # checks and setup
    matchlast = _check_topology(segments, criteria, expert)
    if executor is None:
        executor = ThreadPoolExecutor  # todo: some kind of null executor?
        max_workers = 1
    if max_workers is None: max_workers = util.cpu_count()
    sizes = [len(s.bodyid) for s in segments]
    end = len(segments) - 1
    while end > 1 and (util.bigprod(sizes[end:]) < max_workers or
                       memsize <= 64 * util.bigprod(sizes[:end])): end -= 1
    ntot, chunksize, nchunks = (util.bigprod(x)
                                for x in (sizes, sizes[:end], sizes[end:]))
    if max_samples is not None:
        max_samples = np.clip(chunksize * max_workers, max_samples, ntot)
    every_other = max(1, int(ntot / max_samples)) if max_samples else 1
    nworker = max_workers or util.cpu_count()
    njob = int(np.sqrt(nchunks / every_other) / 128) * nworker
    njob = np.clip(nworker, njob, nchunks)

    actual_ntot = int(ntot / every_other)
    actual_nchunk = int(nchunks / every_other)
    actual_perjob = int(ntot / every_other / njob)
    actual_chunkperjob = int(nchunks / every_other / njob)

    if verbosity >= 0:
        print('tot: {:,} chunksize: {:,} nchunks: {:,} nworker: {} njob: {}'.format(
            ntot, chunksize, nchunks, nworker, njob))
        print('worm/job: {:,} chunk/job: {} sizes={} every_other={}'.format(
            int(ntot / njob), int(nchunks / njob), sizes, every_other))
        print('max_samples: {:,} max_results: {:,}'.format(
            max_samples, max_results))
        print('actual tot:        {:,}'.format(int(actual_ntot)))
        print('actual nchunks:    {:,}'.format(int(actual_nchunk)))
        print('actual worms/job:  {:,}'.format(int(actual_perjob)))
        print('actual chunks/job: {:,}'.format(int(actual_chunkperjob)))

    if njob > 1e9 or nchunks >= 2**63 or every_other >= 2**63:
        print('too big?!?')
        print('    njob', njob)
        print('    nchunks', nchunks, nchunks / 2**63)
        print('    every_other', every_other, every_other / 2**63)
        raise ValueError('system too big')

    # run the stuff
    accumulator = util.WormsAccumulator(
        max_results=max_results, max_tmp_size=1e5)

    # terrible hack... xfering the poses too expensive
    tmp = {s: s.body for seg in segments for s in seg.spliceables}
    for seg in segments:
        for s in seg.spliceables:
            s.body = None  # poses not pickleable...

    with executor(**executor_args) as pool:
        context = (sizes[end:], njob, segments, end, criteria, thresh,
                   matchlast, every_other, max_results)
        args = [range(njob)] + [it.repeat(context)]
        util.tqdm_parallel_map(
            pool=pool,
            function=_grow_chunks,
            accumulator=accumulator,
            map_func_args=args,
            batch_size=nworker * 8,
            unit='K worms',
            ascii=0,
            desc='growing worms',
            unit_scale=int(ntot / njob / 1000 / every_other),
            disable=verbosity < 0
        )

    # put the poses back...
    for seg in segments:
        for s in seg.spliceables:
            s.body = tmp[s]

    # compose and sort results

    result = accumulator.final_result()
    if result is None: return None
    scores, lowidx, lowpos = result

    # scores = np.concatenate([c[0] for c in chunks])
    # order = np.argsort(scores)
    # scores = scores[order]
    # lowidx = np.concatenate([c[1] for c in chunks])[order]
    # lowpos = np.concatenate([c[2] for c in chunks])[order]
    # lowposlist = [lowpos[:, i] for i in range(len(segments))]

    lowposlist = [lowpos[:, i] for i in range(len(segments))]
    score_check = criteria.score(segpos=lowposlist, verbosity=verbosity)
    assert np.allclose(score_check, scores)
    detail = dict(ntot=ntot, chunksize=chunksize, nchunks=nchunks,
                  nworker=nworker, njob=njob, sizes=sizes, end=end)
    return Worms(segments, scores, lowidx, lowpos, criteria, detail)
