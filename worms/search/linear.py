import os, sys, types
from posixpath import expanduser
from time import perf_counter
import numpy as np, numba as nb

from worms.util import jit, InProcessExecutor, PING, Bunch

from worms.vertex import _Vertex, vertex_xform_dtype

from worms.edge import _Edge
from random import random

import concurrent.futures as cf
from worms.search.result import ResultJIT, zero_search_stats
from multiprocessing import cpu_count
from tqdm import tqdm
from time import time, perf_counter as clock
from worms.search.result import remove_duplicate_results, ResultJIT

from worms.util.jitutil import expand_results

@jit
def null_lossfunc(pos, idx, verts):
   return 0.0

def lossfunc_rand_1_in(n):
   @jit
   def func(pos, idx, verts):
      return float(random() * float(n))

   return func

def grow_linear(
   ssdag,
   loss_function=null_lossfunc,
   tolerance=1.0,
   last_bb_same_as=-1,
   parallel=0,
   monte_carlo=0,
   verbosity=0,
   merge_bblock=None,
   lbl="",
   pbar=False,
   pbar_interval=10.0,
   no_duplicate_bases=True,
   max_linear=1000000,
   debug=False,
   **kw,
):
   PING('grow_linear begin')
   kw = Bunch(kw)
   verts = ssdag.verts
   edges = ssdag.edges
   loss_threshold = tolerance
   if last_bb_same_as is None:
      last_bb_same_as = -1
   assert len(verts) > 1
   assert len(verts) == len(edges) + 1
   assert verts[0].dirn[0] == 2
   assert verts[-1].dirn[1] == 2
   for ivertex in range(len(verts) - 1):
      assert verts[ivertex].dirn[1] + verts[ivertex + 1].dirn[0] == 1

   if not isinstance(monte_carlo, (int, float)):
      assert len(monte_carlo) == 1
      monte_carlo = monte_carlo[0]

   # if isinstance(loss_function, types.FunctionType):
   #     if not 'NUMBA_DISABLE_JIT' in os.environ:
   #         loss_function = nb.njit(nogil=True, fastmath=True)

   PING('grow_linear start exe')
   exe = (cf.ThreadPoolExecutor(max_workers=parallel) if parallel else InProcessExecutor())
   # exe = cf.ProcessPoolExecutor(max_workers=parallel) if parallel else InProcessExecutor()
   with exe as pool:
      bb_base = tuple([
         np.array(
            [b.basehash if no_duplicate_bases else 0 for b in bb],
            dtype=np.int64,
         ) for bb in ssdag.bbs
      ])
      verts_pickleable = [v._state for v in verts]
      edges_pickleable = [e._state for e in edges]
      kwargs = dict(
         bb_base=bb_base,
         verts_pickleable=verts_pickleable,
         edges_pickleable=edges_pickleable,
         loss_function=loss_function,
         loss_threshold=loss_threshold,
         last_bb_same_as=last_bb_same_as,
         nresults=0,
         isplice=0,
         splice_position=np.eye(4, dtype=vertex_xform_dtype),
         max_linear=max_linear,
         debug=debug,
         timer=kw.timer,
      )
      futures = list()
      if monte_carlo:
         PING('using monte_carlo')
         kwargs["fn"] = _grow_linear_mc_start
         kwargs["seconds"] = monte_carlo
         kwargs["ivertex_range"] = (0, verts[0].len)
         kwargs["merge_bblock"] = merge_bblock
         kwargs["lbl"] = lbl
         kwargs["verbosity"] = verbosity
         kwargs["pbar"] = pbar
         kwargs["pbar_interval"] = pbar_interval
         njob = cpu_count() if parallel else 1
         for ivert in range(njob):
            kwargs["threadno"] = ivert
            futures.append(pool.submit(**kwargs))
      else:
         PING('NOT using monte_carlo')
         kwargs["fn"] = _grow_linear_start
         nbatch = max(1, int(verts[0].len / 64 / cpu_count()))
         for ivert in range(0, verts[0].len, nbatch):
            ivert_end = min(verts[0].len, ivert + nbatch)
            kwargs["ivertex_range"] = ivert, ivert_end
            futures.append(pool.submit(**kwargs))
      results = list()
      if monte_carlo:
         for f in cf.as_completed(futures):
            results.append(f.result())
      else:
         desc = "linear search " + str(lbl)
         if merge_bblock is None:
            merge_bblock = 0
         fiter = cf.as_completed(futures)
         if pbar:
            fiter = tqdm(
               fiter,
               desc=desc,
               position=merge_bblock + 1,
               mininterval=pbar_interval,
               total=len(futures),
            )
         PING('for f in fiter')
         if 'timer' in kw:
            kw.timer.checkpoint('grow_linear')
         for f in fiter:
            # print('linear.py:grow_linear:f in fiter',flush=True)
            results.append(f.result())
            # print('linear.py:grow_linear:f in fiter DONE',flush=True)
         # print('linear.py:grow_linear for f in fiter DONE', flush=True)
         if 'timer' in kw:
            kw.timer.checkpoint('grow_linear jobsdone')
   tot_stats = zero_search_stats()
   for i in range(len(tot_stats)):
      tot_stats[i][0] += sum([r.stats[i][0] for r in results])

   PING('gather results')
   result = ResultJIT(
      pos=np.concatenate([r.pos for r in results]),
      idx=np.concatenate([r.idx for r in results]),
      err=np.concatenate([r.err for r in results]),
      stats=tot_stats,
   )
   result = remove_duplicate_results(result)
   order = np.argsort(result.err)

   PING('returning ResultJIT')

   return ResultJIT(
      pos=result.pos[order],
      idx=result.idx[order],
      err=result.err[order],
      stats=result.stats,
   )

def _grow_linear_start(
   bb_base,
   verts_pickleable,
   edges_pickleable,
   debug,
   timer,
   **kwargs,
):
   # debug = True

   if debug: print('linear.py:_grow_linear_start begin', flush=True)

   verts = tuple([_Vertex(*vp) for vp in verts_pickleable])
   edges = tuple([_Edge(*ep) for ep in edges_pickleable])
   pos = np.empty(shape=(1024, len(verts), 4, 4), dtype=np.float32)
   idx = np.empty(shape=(1024, len(verts)), dtype=np.int32)
   err = np.empty(shape=(1024, ), dtype=np.float32)
   stats = zero_search_stats()
   result = ResultJIT(pos=pos, idx=idx, err=err, stats=stats)
   bases = np.zeros(len(verts), dtype=np.int64)
   if debug:

      print(f'reslt is {"NONE" if result is None else "OK"}')
      print(f'bbase is {"NONE" if bb_base is None else "OK"}')
      print(f'verts is {"NONE" if verts is None else "OK"}')
      print(f'edges is {"NONE" if edges is None else "OK"}')
      print(f'bases is {"NONE" if bases is None else "OK"}')
      print('loss_function', kwargs['loss_function'] is None)
      print('loss_threshold', kwargs['loss_threshold'] is None)
      print('last_bb_same_as', kwargs['last_bb_same_as'] is None)
      print('nresults', kwargs['nresults'] is None)
      print('max_linear', kwargs['max_linear'] is None)
      print('isplice', kwargs['isplice'] is None)
      print('ivertex_range', kwargs['ivertex_range'] is None)
      print('splice_position', kwargs['splice_position'] is None)
      # with nb.objmode():
      # print('', flush=True)

   # assert 0
   if debug: PING('_grow_linear_start calling _grow_linear_recurse')
   if timer:
      timer.checkpoint('_grow_linear_start')
   # tt = perf_counter()
   nresults, result, _ = _grow_linear_recurse(
      result=result,
      bb_base=bb_base,
      verts=verts,
      edges=edges,
      bases=bases,
      bbidx_prev=-np.ones((len(verts), ), dtype=np.int64),
      expand_results=expand_results,
      **kwargs,
   )
   # print('!!!!!!!!!!!!!!!!!!!!!!!!!', perf_counter() - tt)
   if timer:
      timer.checkpoint('_grow_linear_recurse')
   if debug:
      PING('_grow_linear_start calling _grow_linear_recurse DONE')
      print('nresults', nresults)
      print('result pos', result.pos.shape)
      print('result idx', result.idx.shape)
      print('result err', result.err.shape)
      print('rslt stats', result.stats, flush=True)

   result = ResultJIT(
      result.pos[:nresults],
      result.idx[:nresults],
      result.err[:nresults],
      result.stats,
   )

   if debug: PING('_grow_linear_start DONE', flush=True)
   return result

@jit
def _site_overlap(result, verts, ivertex, nresults, last_bb_same_as):
   # if no 'cyclic' constraint, no checks required
   if last_bb_same_as < 0:
      return False
   i_last_same = result.idx[nresults, last_bb_same_as]
   isite_last_same_in = verts[last_bb_same_as].isite[i_last_same, 0]
   isite_last_same_out = verts[last_bb_same_as].isite[i_last_same, 1]
   # can't reuse same site
   if verts[-1].isite[ivertex, 0] == isite_last_same_in:
      return True
   if verts[-1].isite[ivertex, 0] == isite_last_same_out:
      return True
   return False

@jit
def _last_bb_mismatch(result, verts, ivertex, nresults, last_bb_same_as):
   # if no 'cyclic' constraint, no checks required
   if last_bb_same_as < 0:
      return False
   i_last_same = result.idx[nresults, last_bb_same_as]
   ibblock_last_same = verts[last_bb_same_as].ibblock[i_last_same]
   # last bblock must be same as 'last_bb_same_as'
   if verts[-1].ibblock[ivertex] != ibblock_last_same:
      return True
   return False

# _grow_linear_recurse_jit = jit(_grow_linear_recurse)

def _grow_linear_recurse(
   *,
   result,
   bb_base,
   verts,
   edges,
   loss_function,
   loss_threshold,
   last_bb_same_as,
   nresults,
   max_linear,
   isplice,
   ivertex_range,
   splice_position,
   bases,
   bbidx_prev,
   expand_results,
   debug=False,
):
   """Takes a partially built 'worm' of length isplice and extends them by one based on ivertex_range

    Args:
        result (ResultJIT): accumulated pos, idx, and err
        verts (tuple(_Vertex)*N): Vertices in the linear 'ssdag', store entry/exit geometry
        edges (tuple(_Edge)*(N-1)): Edges in the linear 'ssdag', store allowed splices
        loss_function (jit function): Arbitrary loss function, must be numba-jitable
        loss_threshold (float): only worms with loss <= loss_threshold are put into result
        nresults (int): total number of accumulated results so far
        isplice (int): index of current out-vertex / edge / splice
        ivertex_range (tuple(int, int)): range of ivertex with allowed entry ienter
        splice_position (float32[:4,:4]): rigid body position of splice

    Returns:
        (int, ResultJIT): accumulated pos, idx, and err
    """,

   # with nb.objmode():
   #    print('============ _grow_linear_recurse ===========', flush=True)
   # assert 0

   debug = False
   current_vertex = verts[isplice]
   for ivertex in range(*ivertex_range):
      if not (last_bb_same_as >= 0 and isplice == len(edges)):
         basehash = bb_base[isplice][current_vertex.ibblock[ivertex]]
         if basehash != 0 and np.any(basehash == bases[:isplice]):
            continue
         bases[isplice] = basehash
      if debug: print('ivertex', ivertex)

      result.idx[nresults, isplice] = ivertex
      # assert splice_position.dtype is np.float32, 'splice_position not 32'
      # assert current_vertex.x2orig.dtype is np.float32, 'current_vertex not 32'
      if debug: print('      vertex_position = splice_position @ current_vertex.x2orig[ivertex]')
      vertex_position = splice_position @ current_vertex.x2orig[ivertex]
      # if debug: print('      vertex_position = splice_position @ current_vertex.x2orig[ivertex]')
      if debug: print('      result.pos[nresults, isplice] = vertex_position')
      result.pos[nresults, isplice] = vertex_position
      # if debug: print('      result.pos[nresults, isplice] = vertex_position')
      if isplice == len(edges):
         if debug: print('      isplice == len(edges):')
         # if debug: print('      if isplice == len(edges):')

         # bbidx = np.ones(len(verts), dtype=np.int64)
         bbidx = -np.ones((len(verts), ), dtype=np.int64)
         for isp, vrt in enumerate(verts):
            ivrt = result.idx[nresults, isp]
            bbidx[isp] = vrt.ibblock[ivrt]
         if np.any(bbidx != bbidx_prev):
            if debug: print('linear.py:_grow_linear_recurse current bblocks', bbidx)
            bbidx_prev = bbidx

         if debug: print('         result.stats.total_samples[0] += 1')

         result.stats.total_samples[0] += 1
         if _site_overlap(result, verts, ivertex, nresults, last_bb_same_as):
            # if debug:
            # print('    if _site_overlap(result, verts, ivertex, nresults, last_bb_same_as):')
            continue
         # else:
         # if debug:
         # print('    ELSE _site_overlap(result, verts, ivertex, nresults, last_bb_same_as):')
         result.stats.n_last_bb_same_as[0] += 1
         if debug:
            print('     loss = loss_function(result.pos[nresults], result.idx[nresults], verts)')

         loss = loss_function(result.pos[nresults], result.idx[nresults], verts)

         if debug: print('     DONE loss = loss_function(...)')

         if loss < result.stats.best_score[0]:
            result.stats.best_score[0] = loss
         if result.stats.total_samples[0] % 50000 == 0:
            print(
               'total_samples',
               result.stats.total_samples[0] / 1000,
               'K best',
               result.stats.best_score[0],
            )

         result.err[nresults] = loss
         if loss <= loss_threshold:
            if debug: print('         if loss <= loss_threshold:')
            if nresults >= max_linear:
               if debug: print('            if nresults >= max_linear:')
               return nresults, result, bbidx_prev
            nresults += 1

            if nresults % 10000 == 0:
               print('rnresults', result.stats)

            if debug: print('            result = expand_results(result, nresults)')
            result = expand_results(result, nresults)
      else:
         if debug: print('else isplice == len(edges):')
         next_vertex = verts[isplice + 1]
         next_splicepos = splice_position @ current_vertex.x2exit[ivertex]
         iexit = current_vertex.exit_index[ivertex]
         allowed_entries = edges[isplice].allowed_entries(iexit)
         for ienter in allowed_entries:
            next_ivertex_range = next_vertex.entry_range(ienter)
            if isplice + 1 == len(edges):
               if _last_bb_mismatch(result, verts, next_ivertex_range[0], nresults,
                                    last_bb_same_as):
                  continue
            assert next_ivertex_range[0] >= 0, "ivrt rng err"
            assert next_ivertex_range[1] >= 0, "ivrt rng err"
            assert next_ivertex_range[0] <= next_vertex.len, "ivrt rng err"
            assert next_ivertex_range[1] <= next_vertex.len, "ivrt rng err"
            nresults, result, bbidx_prev = _grow_linear_recurse(
               result=result,
               bb_base=bb_base,
               verts=verts,
               edges=edges,
               loss_function=loss_function,
               loss_threshold=loss_threshold,
               last_bb_same_as=last_bb_same_as,
               nresults=nresults,
               max_linear=max_linear,
               isplice=isplice + 1,
               ivertex_range=next_ivertex_range,
               splice_position=next_splicepos,
               bases=bases,
               bbidx_prev=bbidx_prev,
               expand_results=expand_results,
               debug=debug,
            )
   if debug: print('   return nresults, result')
   return nresults, result, bbidx_prev

def _grow_linear_mc_start(
   *,
   seconds,
   verts_pickleable,
   edges_pickleable,
   threadno,
   pbar,
   lbl,
   verbosity,
   merge_bblock,
   pbar_interval,
   debug,
   timer,
   **kwargs,
):

   # raise NotImplementedError('some features need adding to mc search version')
   tstart = time()
   verts = tuple([_Vertex(*vp) for vp in verts_pickleable])
   edges = tuple([_Edge(*ep) for ep in edges_pickleable])
   pos = np.empty(shape=(1024, len(verts), 4, 4), dtype=np.float32)
   idx = np.empty(shape=(1024, len(verts)), dtype=np.int32)
   err = np.empty(shape=(1024, ), dtype=np.float32)
   stats = zero_search_stats()
   result = ResultJIT(pos=pos, idx=idx, err=err, stats=stats)
   bases = np.zeros(len(verts), dtype=np.int64)
   del kwargs["nresults"]

   if threadno == 0 and pbar:
      desc = "linear search " + str(lbl)
      if merge_bblock is None:
         merge_bblock = 0
      pbar_inst = tqdm(
         desc=desc,
         position=merge_bblock + 1,
         total=seconds,
         mininterval=pbar_interval,
      )
      last = tstart

   nbatch = [1000, 330, 100, 33, 10, 3] + [1] * 99
   nbatch = nbatch[len(edges)] * 10
   nresults = 0
   iter = 0
   ndups = 0
   while time() < tstart + seconds:
      if "pbar_inst" in vars():
         pbar_inst.update(time() - last)
         last = time()
      nresults, result = _grow_linear_mc(
         nbatch,
         result,
         verts,
         edges,
         bases=bases,
         nresults=nresults,
         debug=debug,
         **kwargs,
         # bb_base
         # loss_function
         # loss_threshold
         # last_bb_same_as
         # isplice
         # splice_position
         # max_linear
         # timer
         # ivertex_range
      )

      iter += 1
      # remove duplicates every 10th iter
      if iter % 10 == 0:
         nresults_with_dups = nresults
         uniq_result = ResultJIT(
            idx=result.idx[:nresults],
            pos=result.pos[:nresults],
            err=result.err[:nresults],
            stats=result.stats,
         )
         uniq_result = remove_duplicate_results(uniq_result)
         nresults = len(uniq_result.err)
         result.idx[:nresults] = uniq_result.idx
         result.pos[:nresults] = uniq_result.pos
         result.err[:nresults] = uniq_result.err
         ndups += nresults_with_dups - nresults
         # print(ndups / nresults)

      if nresults >= kwargs["max_linear"]:
         break

   if "pbar_inst" in vars():
      pbar_inst.close()

   result = ResultJIT(
      result.pos[:nresults],
      result.idx[:nresults],
      result.err[:nresults],
      result.stats,
   )
   return result

@jit
def _grow_linear_mc(
   niter,
   result,
   verts,
   edges,
   loss_function,
   loss_threshold,
   last_bb_same_as,
   nresults,
   max_linear,
   isplice,
   ivertex_range,
   splice_position,
   bb_base,
   bases,
   debug,
):

   for i in range(niter):
      nresults, result = _grow_linear_mc_recurse(
         result=result,
         bb_base=bb_base,
         verts=verts,
         edges=edges,
         loss_function=loss_function,
         loss_threshold=loss_threshold,
         last_bb_same_as=last_bb_same_as,
         nresults=nresults,
         max_linear=max_linear,
         isplice=isplice,
         ivertex_range=ivertex_range,
         splice_position=splice_position,
         bases=bases,
         # expand_results=expand_results,
         debug=debug,
      )
   return nresults, result

@jit
def _grow_linear_mc_recurse(
   result,
   bb_base,
   verts,
   edges,
   loss_function,
   loss_threshold,
   last_bb_same_as,
   nresults,
   max_linear,
   isplice,
   ivertex_range,
   splice_position,
   bases,
   debug,
):
   """Takes a partially built 'worm' of length isplice and extends them by one based on ivertex_range

    Args:
        result (ResultJIT): accumulated pos, idx, and err
        verts (tuple(_Vertex)*N): Vertices in the linear 'ssdag', store entry/exit geometry
        edges (tuple(_Edge)*(N-1)): Edges in the linear 'ssdag', store allowed splices
        loss_function (jit function): Arbitrary loss function, must be numba-jitable
        loss_threshold (float): only worms with loss <= loss_threshold are put into result
        nresults (int): total number of accumulated results so far
        isplice (int): index of current out-vertex / edge / splice
        ivertex_range (tuple(int, int)): range of ivertex with allowed entry ienter
        splice_position (float32[:4,:4]): rigid body position of splice

    Returns:
        (int, ResultJIT): accumulated pos, idx, and err
    """

   current_vertex = verts[isplice]
   ivertex = np.random.randint(*ivertex_range)
   if not (last_bb_same_as >= 0 and isplice == len(edges)):
      basehash = bb_base[isplice][current_vertex.ibblock[ivertex]]
      if basehash != 0 and np.any(basehash == bases[:isplice]):
         return nresults, result
      bases[isplice] = basehash
   result.idx[nresults, isplice] = ivertex
   vertex_position = splice_position @ current_vertex.x2orig[ivertex]
   result.pos[nresults, isplice] = vertex_position
   if isplice == len(edges):
      result.stats.total_samples[0] += 1
      if _site_overlap(result, verts, ivertex, nresults, last_bb_same_as):
         return nresults, result
      result.stats.n_last_bb_same_as[0] += 1
      loss = loss_function(result.pos[nresults], result.idx[nresults], verts)
      result.err[nresults] = loss
      if loss < result.stats.best_score[0]:
         result.stats.best_score[0] = loss
      if loss <= loss_threshold:
         if nresults >= max_linear:
            return nresults, result
         nresults += 1
         result = expand_results(result, nresults)

      if result.stats.total_samples[0] % 10000 == 0:
         print(
            'total_samples',
            result.stats.total_samples[0] / 1000,
            'K best',
            result.stats.best_score[0],
         )

   else:
      next_vertex = verts[isplice + 1]
      next_splicepos = splice_position @ current_vertex.x2exit[ivertex]
      iexit = current_vertex.exit_index[ivertex]
      allowed_entries = edges[isplice].allowed_entries(iexit)
      if len(allowed_entries) == 0:
         return nresults, result
      iskip = max(1, len(allowed_entries) / 100)
      istart = np.random.randint(0, iskip)
      for ienter in allowed_entries[istart:None:iskip]:
         # for ienter in allowed_entries:
         next_ivertex_range = next_vertex.entry_range(ienter)
         if isplice + 1 == len(edges):
            if _last_bb_mismatch(result, verts, next_ivertex_range[0], nresults, last_bb_same_as):
               continue
         assert next_ivertex_range[0] >= 0, "ivrt rng err"
         assert next_ivertex_range[1] >= 0, "ivrt rng err"
         assert next_ivertex_range[0] <= next_vertex.len, "ivrt rng err"
         assert next_ivertex_range[1] <= next_vertex.len, "ivrt rng err"
         nresults, result = _grow_linear_mc_recurse(
            result=result,
            bb_base=bb_base,
            verts=verts,
            edges=edges,
            loss_function=loss_function,
            loss_threshold=loss_threshold,
            last_bb_same_as=last_bb_same_as,
            nresults=nresults,
            max_linear=max_linear,
            isplice=isplice + 1,
            ivertex_range=next_ivertex_range,
            splice_position=next_splicepos,
            bases=bases,
            # expand_results=expand_results,
            debug=debug,
         )
   return nresults, result
