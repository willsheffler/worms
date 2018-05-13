"""search stuff
"""

import sys
import os
import _pickle as pickle
import itertools as it
import numpy as np
from collections import defaultdict
from xbin import XformBinner
from homog import hinv, hrot
from concurrent.futures import ProcessPoolExecutor
from worms.segments import Segment, Segments, Worms
from worms.criteria import CriteriaList, Cyclic, WormCriteria
from worms import util


class SimpleAccumulator:
    """TODO: Summary

    Attributes:
        lowidx (TYPE): Description
        lowpos (TYPE): Description
        max_results (TYPE): Description
        max_tmp_size (TYPE): Description
        scores (TYPE): Description
        temporary (list): Description
    """

    def __init__(self, max_results=1000000, max_tmp_size=1024):
        """TODO: Summary

        Args:
            max_results (int, optional): Description
            max_tmp_size (int, optional): Description
        """
        self.max_tmp_size = max_tmp_size
        self.max_results = max_results
        self.temporary = []

    def checkpoint(self):
        """TODO: Summary
        """
        if len(self.temporary) is 0: return
        if hasattr(self, 'scores'):
            sc, li, lp = [self.scores], [self.lowidx], [self.lowpos]
        else:
            sc, li, lp = [], [], []
        scores = np.concatenate([x[0] for x in self.temporary] + sc)
        lowidx = np.concatenate([x[1] for x in self.temporary] + li)
        lowpos = np.concatenate([x[2] for x in self.temporary] + lp)
        order = np.argsort(scores)
        self.scores = scores[order[:self.max_results]]
        self.lowidx = lowidx[order[:self.max_results]]
        self.lowpos = lowpos[order[:self.max_results]]
        self.temporary = []

    def accumulate(self, gen):
        """TODO: Summary

        Args:
            gen (TYPE): Description

        Yields:
            TYPE: Description
        """
        for future in gen:
            result = future.result()
            if result is not None:
                self.temporary.append(result)
                if len(self.temporary) >= self.max_tmp_size:
                    self.checkpoint()
            yield None

    def final_result(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        self.checkpoint()
        try:
            return self.scores, self.lowidx, self.lowpos
        except AttributeError:
            return None


class MakeXIndexAccumulator:
    """TODO: Summary

    Attributes:
        binner (TYPE): Description
        from_seg (TYPE): Description
        max_tmp_size (TYPE): Description
        sizes (TYPE): Description
        thresh (TYPE): Description
        tmp (list): Description
        to_seg (TYPE): Description
        xindex (TYPE): Description
    """

    def __init__(self,
                 sizes,
                 thresh=1,
                 from_seg=0,
                 to_seg=-1,
                 max_tmp_size=1024,
                 cart_resl=None,
                 ori_resl=None):
        """TODO: Summary

        Args:
            sizes (TYPE): Description
            thresh (int, optional): Description
            from_seg (int, optional): Description
            to_seg (TYPE, optional): Description
            max_tmp_size (int, optional): Description
            cart_resl (None, optional): Description
            ori_resl (None, optional): Description
        """
        self.sizes = sizes
        self.thresh = thresh
        self.max_tmp_size = max_tmp_size
        self.from_seg = from_seg
        self.to_seg = to_seg
        self.tmp = []
        self.binner = XformBinner(cart_resl, ori_resl)

        self.xindex = defaultdict(list)
        # self.xindex = dict()

    def checkpoint(self):
        """TODO: Summary
        """
        print('MakeXIndexAccumulator checkpoint', end='')
        sys.stdout.flush()
        if len(self.tmp) is 0: return
        sc = np.concatenate([x[0] for x in self.tmp])
        indices = np.concatenate([x[1] for x in self.tmp])[sc <= self.thresh]
        assert np.all(indices < self.sizes)
        positions = np.concatenate([x[2] for x in self.tmp])[sc <= self.thresh]
        from_pos = positions[:, self.from_seg]
        to_pos = positions[:, self.to_seg]
        xtgt = hinv(from_pos) @ to_pos
        bin_idx = self.binner.get_bin_index(xtgt)

        for k, v in zip(bin_idx, indices):
            self.xindex[k].append(v)
        # self.xindex = {**{k: v for k, v in zip(bin_idx, indices)},
        # **self.xindex}

        # print('IndexAcculator checkpoint, xindex size:', len(self.xindex))
        self.tmp = []
        print('done, xindex size:', len(self.xindex),
              sum(len(v) for v in self.xindex.values()))
        sys.stdout.flush()

    def accumulate(self, gen):
        """TODO: Summary

        Args:
            gen (TYPE): Description

        Yields:
            TYPE: Description
        """
        for future in gen:
            result = future.result()
            if result is not None:
                self.tmp.append(result)
                if len(self.tmp) >= self.max_tmp_size:
                    self.checkpoint()
            yield None

    def final_result(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        self.checkpoint()
        return self.xindex, self.binner


class XIndexedCriteria(WormCriteria):
    """TODO: Summary

    Attributes:
        binner (TYPE): Description
        cyclic_xform (TYPE): Description
        from_seg (TYPE): Description
        xindex_set (TYPE): Description
    """

    def __init__(self, xindex, binner, nfold, from_seg=-1):
        """TODO: Summary

        Args:
            xindex (TYPE): Description
            binner (TYPE): Description
            nfold (TYPE): Description
            from_seg (TYPE, optional): Description
        """
        self.xindex_set = set(xindex.keys())
        self.binner = binner
        self.from_seg = from_seg
        self.cyclic_xform = hrot([0, 0, 1], 360 / nfold)
        # global GLOBAL_xindex_set
        # GLOBAL_xindex_set = self.xindex_set

    def get_xform_commutator(self, from_pos, to_pos):
        """TODO: Summary

        Args:
            from_pos (TYPE): Description
            to_pos (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.linalg.inv(from_pos) @ to_pos

    def is_in_xindex_set(self, idxary):
        """TODO: Summary

        Args:
            idxary (TYPE): Description

        Returns:
            TYPE: Description
        """
        is_in = np.ones(idxary.size, dtype='f') * 999999.
        for i, idx in enumerate(idxary.flat):
            if idx in self.xindex_set:
                is_in[i] = 0
        return is_in.reshape(idxary.shape)

    def score(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            kw: passthru args
        Returns:
            TYPE: Description
        """
        from_pos = segpos[self.from_seg]
        to_pos = self.cyclic_xform @ from_pos
        xtgt = self.get_xform_commutator(from_pos, to_pos)
        bin_idx = self.binner.get_bin_index(xtgt)
        return self.is_in_xindex_set(bin_idx)

    def alignment(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            kw: passthru args
        Returns:
            TYPE: Description
        """
        return np.eye(4)


class XIndexedAccumulator:
    """TODO: Summary

    Attributes:
        binner (TYPE): Description
        cyclic_xform (TYPE): Description
        from_seg (TYPE): Description
        head (TYPE): Description
        lowidx (TYPE): Description
        max_results (TYPE): Description
        max_tmp_size (TYPE): Description
        segments (TYPE): Description
        splitpoint (TYPE): Description
        tail (TYPE): Description
        temporary (list): Description
        to_seg (TYPE): Description
        xindex (TYPE): Description
    """

    def __init__(self,
                 segments,
                 tail,
                 splitpoint,
                 head,
                 xindex,
                 binner,
                 nfold,
                 from_seg,
                 to_seg,
                 max_tmp_size=1024,
                 max_results=100000):
        """TODO: Summary

        Args:
            segments (TYPE): Description
            tail (TYPE): Description
            splitpoint (TYPE): Description
            head (TYPE): Description
            xindex (TYPE): Description
            binner (TYPE): Description
            nfold (TYPE): Description
            from_seg (TYPE): Description
            to_seg (TYPE): Description
            max_tmp_size (int, optional): Description
            max_results (int, optional): Description
        """
        self.segments = segments
        self.tail = tail
        self.splitpoint = splitpoint
        self.head = head
        self.xindex = xindex
        self.binner = binner
        self.from_seg = from_seg
        self.to_seg = to_seg
        self.cyclic_xform = hrot([0, 0, 1], 360 / nfold)
        self.max_tmp_size = max_tmp_size
        self.max_results = max_results
        self.temporary = []

    def checkpoint(self):
        """TODO: Summary
        """
        if len(self.temporary) is 0: return
        ntmp = sum(len(tmp[0]) for tmp in self.temporary)
        print('XIndexedAccumulator checkpoint... ncandidates:', ntmp, end=' ')
        sys.stdout.flush()
        if hasattr(self, 'scores'):
            sc, li, lp = [self.scores], [self.lowidx], [self.lowpos]
        else:
            sc, li, lp = [], [], []
        scores = np.concatenate([x[0] for x in self.temporary])
        if scores.shape[0] is 0: return
        assert np.all(scores == 0)
        lowidx = np.concatenate([x[1] for x in self.temporary])
        lowpos = np.concatenate([x[2] for x in self.temporary])
        scores = scores[:self.max_results]
        lowpos = lowpos[:self.max_results]
        lowidx = lowidx[:self.max_results]
        from_pos = lowpos[:, -1]
        to_pos = self.cyclic_xform @ from_pos
        xtgt = hinv(from_pos) @ to_pos
        bin_idx = self.binner.get_bin_index(xtgt)

        # head_idx = np.stack([self.xindex[i] for i in bin_idx])
        lowidxtmp, headidxtmp = [], []
        for i, b in enumerate(bin_idx):
            for headidx in self.xindex[b]:
                lowidxtmp.append(lowidx[i])
                headidxtmp.append(headidx)
        lowidx = np.stack(lowidxtmp)
        head_idx = np.stack(headidxtmp)

        join_idx, valid = self.segments[self.splitpoint].merge_idx(
            self.tail[-1], lowidx[:, -1], self.head[0], head_idx[:, 0])
        lowidx = lowidx[valid][join_idx >= 0]
        head_idx = head_idx[valid][join_idx >= 0]
        join_idx = join_idx[join_idx >= 0]

        # join_idx = self.segments[self.splitpoint].merge_idx_slow(
        #     self.tail[-1], lowidx[:, -1],
        #     self.head[0], head_idx[:, 0])
        # lowidx = lowidx[join_idx >= 0]
        # head_idx = head_idx[join_idx >= 0]
        # join_idx = join_idx[join_idx >= 0]
        lowidx = np.concatenate(
            [lowidx[:, :-1], join_idx[:, None], head_idx[:, 1:]], axis=1)

        ifrom, ito = lowidx[:, self.from_seg], lowidx[:, self.to_seg]
        site1 = self.segments[self.from_seg].entrysiteid[ifrom]
        site2 = self.segments[self.from_seg].exitsiteid[ifrom]
        site3 = self.segments[self.to_seg].entrysiteid[ito]
        ok = (site1 != site2) * (site1 != site3) * (site2 != site3)
        # print('!!!!!!!!', np.sum(ok), ok.shape)
        # print('site1', *['%6i' % np.sum(site1 == i) for i in range(10)])
        # print('site2', *['%6i' % np.sum(site2 == i) for i in range(10)])
        # print('site3', *['%6i' % np.sum(site3 == i) for i in range(10)])
        lowidx = lowidx[ok]

        if hasattr(self, 'lowidx'):
            self.lowidx = np.concatenate([self.lowidx, lowidx])
        else:
            self.lowidx = lowidx
        self.temporary = []
        print('done, total pre-err =', len(self.lowidx))
        sys.stdout.flush()

    def accumulate(self, gen):
        """TODO: Summary

        Args:
            gen (TYPE): Description

        Yields:
            TYPE: Description
        """
        for future in gen:
            result = future.result()
            if result is not None:
                self.temporary.append(result)
                if len(self.temporary) >= self.max_tmp_size:
                    self.checkpoint()
            yield None

    def final_result(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        self.checkpoint()
        try:
            return self.lowidx
        except AttributeError:
            return None


def _get_chunk_end_seg(sizes, max_workers, memsize):
    """TODO: Summary

    Args:
        sizes (TYPE): Description
        max_workers (TYPE): Description
        memsize (TYPE): Description

    Returns:
        TYPE: Description
    """
    end = len(sizes) - 1
    while end > 1 and (util.bigprod(sizes[end:]) < max_workers
                       or memsize <= 64 * util.bigprod(sizes[:end])):
        end -= 1
    return end


def grow(segments,
         criteria,
         *,
         thresh=2,
         expert=0,
         memsize=1e6,
         executor=None,
         executor_args=None,
         max_workers=None,
         verbosity=2,
         chunklim=None,
         max_samples=int(1e12),
         max_results=int(1e6),
         cart_resl=2.0,
         ori_resl=10.0,
         xindex_cache_file=None):
    """TODO: Summary

    Args:
        segments (TYPE): Description
        criteria (TYPE): Description
        thresh (int, optional): Description
        expert (int, optional): Description
        memsize (float, optional): Description
        executor (None, optional): Description
        executor_args (None, optional): Description
        max_workers (None, optional): Description
        verbosity (int, optional): Description
        chunklim (None, optional): Description
        max_samples (TYPE, optional): Description
        max_results (TYPE, optional): Description
        cart_resl (float, optional): Description
        ori_resl (float, optional): Description
        xindex_cache_file (None, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if True:  # setup
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        if isinstance(segments, list):
            segments = Segments(segments)
        # if isinstance(executor, (ProcessPoolExecutor, ThreadPoolExecutor)):
        # raise ValueError('please use dask.distributed executor')
        if verbosity > 0:
            print('grow, from', criteria.from_seg, 'to', criteria.to_seg)
            for i, seg in enumerate(segments):
                print(' segment', i, 'enter:', seg.entrypol, 'exit:',
                      seg.exitpol)
                for sp in seg.spliceables:
                    print('   ', sp)
        elif verbosity == 0:
            print('grow, nseg:', len(segments))
        if verbosity > 2:
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

    max_results = int(max_results)

    if executor is None:
        executor = util.InProcessExecutor
        max_workers = 1
    if max_workers is None: max_workers = util.cpu_count()
    nworker = max_workers or util.cpu_count()

    if criteria.origin_seg is None:

        matchlast = _check_topology(segments, criteria, expert)
        sizes = [len(s) for s in segments]
        end = _get_chunk_end_seg(sizes, max_workers, memsize)
        ntot, chunksize, nchunks = (util.bigprod(x)
                                    for x in (sizes, sizes[:end], sizes[end:]))
        if max_samples is not None:
            max_samples = np.clip(chunksize * max_workers, max_samples, ntot)
        every_other = max(1, int(ntot / max_samples)) if max_samples else 1
        njob = int(np.sqrt(nchunks / every_other) / 128) * nworker
        njob = np.clip(nworker, njob, nchunks)

        actual_ntot = int(ntot / every_other)
        actual_nchunk = int(nchunks / every_other)
        actual_perjob = int(ntot / every_other / njob)
        actual_chunkperjob = int(nchunks / every_other / njob)
        if verbosity >= 0:
            print(
                'tot: {:,} chunksize: {:,} nchunks: {:,} nworker: {} njob: {}'.
                format(ntot, chunksize, nchunks, nworker, njob))
            print(
                'worm/job: {:,} chunk/job: {} sizes={} every_other={}'.format(
                    int(ntot / njob), int(nchunks / njob), sizes, every_other))
            print('max_samples: {:,} max_results: {:,}'.format(
                max_samples, max_results))
            print('actual tot:        {:,}'.format(int(actual_ntot)))
            print('actual nchunks:    {:,}'.format(int(actual_nchunk)))
            print('actual worms/job:  {:,}'.format(int(actual_perjob)))
            print('actual chunks/job: {:,}'.format(int(actual_chunkperjob)))
        _grow_args = dict(
            executor=executor,
            executor_args=executor_args,
            njob=njob,
            end=end,
            thresh=thresh,
            matchlast=matchlast,
            every_other=every_other,
            max_results=max_results,
            nworker=nworker,
            verbosity=verbosity)
        if njob > 1e9 or nchunks >= 2**63 or every_other >= 2**63:
            print('too big?!?')
            print('    njob', njob)
            print('    nchunks', nchunks, nchunks / 2**63)
            print('    every_other', every_other, every_other / 2**63)
            raise ValueError('system too big')
        accum = SimpleAccumulator(max_results=max_results, max_tmp_size=1e5)
        _grow(segments, criteria, accum, **_grow_args)
        result = accum.final_result()
        if result is None: return None
        scores, lowidx, lowpos = result
        lowposlist = [lowpos[:, i] for i in range(len(segments))]
        score_check = criteria.score(segpos=lowposlist, verbosity=verbosity)
        assert np.allclose(score_check, scores)
        detail = dict(
            ntot=ntot,
            chunksize=chunksize,
            nchunks=nchunks,
            nworker=nworker,
            njob=njob,
            sizes=sizes,
            end=end)

    else:  # hash-based protocol...

        assert len(criteria) is 1
        _check_topology(segments, criteria, expert)

        splitpoint = criteria.from_seg
        tail, head = segments.split_at(splitpoint)

        print('HASH PROTOCOL splitting at segment', splitpoint)
        print('    full:', [len(s) for s in segments])

        headsizes = [len(s) for s in head]
        headend = _get_chunk_end_seg(headsizes, max_workers, memsize)
        ntot, chunksize, nchunks = (util.bigprod(x)
                                    for x in (headsizes, headsizes[:headend],
                                              headsizes[headend:]))
        if max_samples is not None:
            max_samples = np.clip(chunksize * max_workers, max_samples, ntot)
        every_other = max(1, int(ntot / max_samples)) if max_samples else 1
        njob = int(np.sqrt(nchunks / every_other) / 16 / nworker) * nworker
        njob = np.clip(nworker, njob, nchunks)
        _grow_args = dict(
            executor=executor,
            executor_args=executor_args,
            njob=njob,
            end=headend,
            thresh=thresh,
            matchlast=0,
            every_other=every_other,
            max_results=1e9,
            nworker=nworker,
            verbosity=verbosity)
        t1 = 0
        if xindex_cache_file and os.path.exists(xindex_cache_file):
            print('!' * 100)
            print('reading xindex, xbinner from', xindex_cache_file)
            xindex, binner = pickle.load(open(xindex_cache_file, 'rb'))
        else:
            # if 1:
            accum1 = MakeXIndexAccumulator(
                headsizes,
                from_seg=0,
                to_seg=-1,
                cart_resl=cart_resl,
                ori_resl=ori_resl)
            headcriteria = Cyclic(
                criteria[0].nfold,
                from_seg=0,
                to_seg=-1,
                tol=criteria[0].tol * 2.0,
                lever=criteria[0].lever)
            print('STEP ONE: growing head into xindex')
            print('    ntot            {:,}'.format(ntot))
            print('    headsizes       {}'.format(headsizes))
            print('    headend         {:,}'.format(headend))
            print('    njob            {:,}'.format(njob))
            print('    nchunks         {:,}'.format(nchunks))
            print('    chunksize       {:,}'.format(chunksize))
            print('    thresh          {:,}'.format(thresh))
            print('    matchlast       {:,}'.format(0))
            print('    every_other     {:,}'.format(every_other))
            print('    max_results     {:,}'.format(max_results))
            print('    nworker         {:,}'.format(nworker))
            print('    act. ntot       {:,}'.format(int(ntot / every_other)))
            print('    act. nchunks    {:,}'.format(
                int(nchunks / every_other)))
            print('    act. worms/job  {:,}'.format(
                int(ntot / every_other / njob)))
            print('    act. chunks/job {:,}'.format(
                int(nchunks / every_other / njob)))

            import time
            t1 = time.time()
            _grow(head, headcriteria, accum1, **_grow_args)
            xindex, binner = accum1.final_result()
            t1 = time.time() - t1
            print('!' * 100)
            print("TIME PHASE ONE", t1)
            print('!' * 100)

            if xindex_cache_file:
                print('!' * 100)
                print('dumping xindex to', xindex_cache_file)
                print('!' * 100)
                pickle.dump((xindex, binner), open(xindex_cache_file, 'wb'))

        ################### PHASE TWO ####################

        tailcriteria = XIndexedCriteria(
            xindex, binner, criteria[0].nfold, from_seg=-1)
        accum2 = XIndexedAccumulator(
            segments,
            tail,
            splitpoint,
            head,
            xindex,
            binner,
            criteria[0].nfold,
            from_seg=criteria.from_seg,
            to_seg=criteria.to_seg,
            max_results=max_results * 200)

        tailsizes = [len(s) for s in tail]
        tailend = _get_chunk_end_seg(tailsizes, max_workers, memsize)
        ntot, chunksize, nchunks = (util.bigprod(x)
                                    for x in (tailsizes, tailsizes[:tailend],
                                              tailsizes[tailend:]))
        if max_samples is not None:
            max_samples = np.clip(chunksize * max_workers, max_samples, ntot)
        every_other = max(1,
                          int(ntot / max_samples * 20)) if max_samples else 1
        njob = int(np.ceil(np.sqrt(nchunks / every_other) / 32 / nworker))
        njob = np.clip(nworker, njob * nworker, nchunks)

        _grow_args = dict(
            executor=executor,
            executor_args=executor_args,
            njob=njob,
            end=tailend,
            thresh=thresh,
            matchlast=None,
            every_other=every_other,
            max_results=max_results,
            nworker=nworker,
            verbosity=verbosity)

        print('STEP TWO: using xindex, nbins: {:,} nentries: {:,}'.format(
            len(xindex), sum(len(x) for x in xindex.values())))
        print('    ntot            {:,}'.format(ntot))
        print('    tailsizes       {}'.format(tailsizes))
        print('    tailend         {:,}'.format(tailend))
        print('    njob            {:,}'.format(njob))
        print('    nchunks         {:,}'.format(nchunks))
        print('    chunksize       {:,}'.format(chunksize))
        print('    thresh          {:,}'.format(thresh))
        print('    matchlast       None')
        print('    every_other     {:,}'.format(every_other))
        print('    max_results     {:,}'.format(max_results))
        print('    nworker         {:,}'.format(nworker))
        print('    act. ntot       {:,}'.format(int(ntot / every_other)))
        print('    act. nchunks    {:,}'.format(int(nchunks / every_other)))
        print('    act. worms/job  {:,}'.format(
            int(ntot / every_other / njob)))
        print('    act. chunks/job {:,}'.format(
            int(nchunks / every_other / njob)))
        print('    executor       ', type(executor()))

        import time
        t2 = time.time()

        _grow(tail, tailcriteria, accum2, **_grow_args)
        # import cProfile
        # cProfile.runctx('_grow(tail, tailcriteria, accum2, **_grow_args)',
        #                 locals(), globals(), 'grow2.stats')
        # import pstats
        # pst = pstats.Stats('grow2.stats')
        # pst.strip_dirs().sort_stats('time').print_stats(20)
        lowidx = accum2.final_result()
        t2 = time.time() - t2

        print('!' * 100)
        print("TIME PHASE ONE", t1)
        print("TIME PHASE TWO", t2)
        # print('   best 28 cores 1608.94K/s small 1min job 681k/.s')
        print('!' * 100)

        if lowidx is None:
            print('grow: no results')
            return

        print('refold segments')
        lowpos = _refold_segments(segments, lowidx)
        lowposlist = [lowpos[:, i] for i in range(len(segments))]
        print('score refolded segments')
        scores = criteria.score(segpos=lowposlist, verbosity=verbosity)
        print('organize results')
        nlow = sum(scores <= thresh)
        order = np.argsort(scores)[:nlow]
        scores = scores[order]
        lowpos = lowpos[order]
        lowidx = lowidx[order]
        detail = dict(
            ntot=ntot,
            chunksize=chunksize,
            nchunks=nchunks,
            nworker=nworker,
            njob=njob,
            sizes=tailsizes,
            end=tailend)

    return Worms(segments, scores, lowidx, lowpos, criteria, detail)


def _refold_segments(segments, lowidx):
    """TODO: Summary

    Args:
        segments (TYPE): Description
        lowidx (TYPE): Description

    Returns:
        TYPE: Description
    """
    pos = np.zeros_like(lowidx, dtype='(4,4)f') + np.eye(4)
    end = np.eye(4)
    for i, seg in enumerate(segments):
        pos[:, i] = end @ seg.x2orgn[lowidx[:, i]]
        end = end @ seg.x2exit[lowidx[:, i]]
    return pos


def _chain_xforms(segments):
    """TODO: Summary

    Args:
        segments (TYPE): Description

    Returns:
        TYPE: Description
    """
    x2exit = [s.x2exit for s in segments]
    x2orgn = [s.x2orgn for s in segments]
    fullaxes = (np.newaxis, ) * (len(x2exit) - 1)
    xconn = [
        x2exit[0][fullaxes],
    ]
    xbody = [
        x2orgn[0][fullaxes],
    ]
    for iseg in range(1, len(x2exit)):
        fullaxes = (slice(None), ) + (np.newaxis, ) * iseg
        xconn.append(xconn[iseg - 1] @ x2exit[iseg][fullaxes])
        xbody.append(xconn[iseg - 1] @ x2orgn[iseg][fullaxes])
    perm = list(range(len(xbody) - 1, -1, -1)) + [len(xbody), len(xbody) + 1]
    xbody = [np.transpose(x, perm) for x in xbody]
    xconn = [np.transpose(x, perm) for x in xconn]
    return xbody, xconn


__print_best = False
__best_score = 9e9


def _grow_chunk(samp, segpos, conpos, context):
    """TODO: Summary

    Args:
        samp (TYPE): Description
        segpos (TYPE): Description
        conpos (TYPE): Description
        context (TYPE): Description

    Returns:
        TYPE: Description
    """
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
            idx = (slice(None), ) * ML + (allowed, )
            segpos = segpos[:ML] + [x[idx] for x in segpos[ML:]]
            conpos = conpos[:ML] + [x[idx] for x in conpos[ML:]]
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
        ilow = ilow0[:iseg + 1] + (0, ) * (segpos[0].ndim - 2 - (iseg + 1))
        lowpostmp.append(segpos[iseg][ilow])
    ilow1 = (ilow0 if (ML is None or ML >= ndimchunk) else
             ilow0[:ML] + (idxmap[ilow0[ML]], ) + ilow0[ML + 1:])
    return score[ilow0], np.array(ilow1 + sampidx).T, np.stack(lowpostmp, 1)


def _grow_chunks(ijob, context):
    """TODO: Summary

    Args:
        ijob (TYPE): Description
        context (TYPE): Description

    Returns:
        TYPE: Description
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    sampsizes, njob, segments, end, _, _, _, every_other, max_results = context
    samples = list(util.MultiRange(sampsizes)[ijob::njob * every_other])
    segpos, connpos = _chain_xforms(segments[:end])  # common data
    args = [samples, it.repeat(segpos), it.repeat(connpos), it.repeat(context)]
    chunks = list(map(_grow_chunk, *args))
    chunks = [c for c in chunks if c is not None]
    if not chunks: return None
    scores = np.concatenate([x[0] for x in chunks])
    lowidx = np.concatenate([x[1] for x in chunks])
    lowpos = np.concatenate([x[2] for x in chunks])
    order = np.argsort(scores)
    return [
        scores[order[:int(max_results)]], lowidx[order[:int(max_results)]],
        lowpos[order[:int(max_results)]]
    ]


def _check_topology(segments, criteria, expert=False):
    """TODO: Summary

    Args:
        segments (TYPE): Description
        criteria (TYPE): Description
        expert (bool, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if segments[0].entrypol is not None:
        raise ValueError('beginning of worm cant have entry')
    if segments[-1].exitpol is not None:
        raise ValueError('end of worm cant have exit')
    for a, b in zip(segments[:-1], segments[1:]):
        if not (a.exitpol and b.entrypol and a.exitpol != b.entrypol):
            raise ValueError('incompatible exit->entry polarity: ' + str(
                a.exitpol) + '->' + str(b.entrypol) + ' on segment pair: ' +
                             str((segments.index(a), segments.index(b))))
    matchlast = criteria.last_body_same_as
    if matchlast is not None and not expert and (
            not segments[matchlast].same_bodies_as(segments[-1])):
        raise ValueError("segments[matchlast] not same as segments[-1], " +
                         "if you're sure, pass expert=True")
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
                    beg.min_sites[pol])
                if not expert: raise ValueError(msg)
                print("WARNING:", msg)
    return matchlast


def _grow(segments, criteria, accumulator, **kw):
    """TODO: Summary

    Args:
        segments (TYPE): Description
        criteria (TYPE): Description
        accumulator (TYPE): Description
        kw: passthru args    """
    # terrible hack... xfering the poses too expensive
    tmp = {
        spl: (spl.body, spl.chains)
        for seg in segments for spl in seg.spliceables
    }
    for seg in segments:
        for spl in seg.spliceables:
            spl.body, spl.chains = None, None  # poses not pickleable...

    sizes = [len(s) for s in segments]
    ntot = util.bigprod(sizes)
    with kw['executor'](**kw['executor_args']) as pool:
        context = (sizes[kw['end']:], kw['njob'], segments, kw['end'],
                   criteria, kw['thresh'], kw['matchlast'], kw['every_other'],
                   kw['max_results'])
        args = [range(kw['njob'])] + [it.repeat(context)]
        util.tqdm_parallel_map(
            pool=pool,
            function=_grow_chunks,
            accumulator=accumulator,
            map_func_args=args,
            batch_size=kw['nworker'] * 8,
            unit='K worms',
            ascii=0,
            desc='growing worms',
            unit_scale=int(ntot / kw['njob'] / 1000 / kw['every_other']),
            disable=kw['verbosity'] < 0)

    # put the poses back...
    for seg in segments:
        for spl in seg.spliceables:
            spl.body, spl.chains = tmp[spl]
