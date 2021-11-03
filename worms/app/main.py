import sys
import os
import pickle
from time import time
import concurrent.futures as cf
import traceback

import worms
from worms.util import Bunch

import blosc
# from xbin import gu_xbin_indexer, numba_xbin_indexer
from worms import PING

from worms.cli import build_worms_setup_from_cli_args
from worms.ssdag import simple_search_dag

from worms.util import run_and_time
from worms import util
from worms.filters.clash import prune_clashes
from worms.filters.geometry import check_geometry

_global_shared_ssdag = None

def worms_main(argv):

   # from worms.homog import rand_xform, numba_axis_angle_cen, hrot, axis_ang_cen_of, numba_hrot
   # x = rand_xform().astype('f4')
   # print(x.shape)
   # axis, angle, cen = numba_axis_angle_cen(x)
   # axis0, angle0, cen0 = axis_ang_cen_of(x)
   # print(axis)
   # print(axis0)
   # print(angle)
   # print(angle0)
   # print(cen)
   # print(cen0)
   # print(numba_hrot(axis, angle, cen))
   # assert 0

   tstart = time()

   blosc.set_releasegil(True)

   criteria_list, kw = build_worms_setup_from_cli_args(argv)

   # try:
   if True:
      construct_global_ssdag_and_run(criteria_list, kw)
   # except Exception as e:
   #    bbdb = kw.database[0]
   #    bbdb.clear()
   #    raise e

   print('worms_main done, time:', time() - tstart)

def construct_global_ssdag_and_run(
   criteria_list,
   kw,
):
   print("construct_global_ssdag_and_run,", len(criteria_list), "criteria, args:")
   orig_output_prefix = kw.output_prefix
   kw.timer.checkpoint('top of construct_global_ssdag_and_run')
   log = list()

   for icrit, criteria in enumerate(criteria_list):
      if len(criteria_list) > 1:
         assert len(criteria_list) is len(kw["config_file"])
         name = os.path.basename(kw["config_file"][icrit])
         name = name.replace(".config", "")
         kw.output_prefix = orig_output_prefix + "_" + name
      print("================== start job", icrit, "======================")
      print("output_prefix:", kw.output_prefix)
      print("criteria:", criteria)
      print("bbspec:", criteria.bbspec, flush=True)

      if kw.precache_splices:  # default
         PING("precaching splices")
         merge_bblock = kw.merge_bblock
         del kw.merge_bblock
         ssd = simple_search_dag(
            criteria,
            merge_bblock=None,
            precache_only=True,
            **kw,
         )
         kw.bbs = ssd.bblocks
         if kw.only_bblocks:
            assert len(kw.bbs) is len(kw.only_bblocks)
            for i, bb in enumerate(kw.bbs):
               kw.bbs[i] = [bb[kw.only_bblocks[i]]]
            print("modified bblock numbers (--only_bblocks)")
            print("   ", [len(b) for b in kw.bbs])
         kw.merge_bblock = merge_bblock
         if kw["precache_splices_and_quit"]:
            return Bunch(log=log)
      kw.timer.checkpoint('precache_splices')

      global _global_shared_ssdag
      if "bbs" in kw and (len(kw.bbs) > 2 or kw.bbs[0] is not kw.bbs[1]):
         merge_bblock = kw.merge_bblock
         del kw.merge_bblock
         ssd = simple_search_dag(criteria, merge_bblock=0, print_edge_summary=True, **kw)
         _global_shared_ssdag = ssd.ssdag
         assert _global_shared_ssdag is not None
         kw.merge_bblock = merge_bblock
         PING("memuse for global _global_shared_ssdag:")
         _global_shared_ssdag.report_memory_use()
         assert _global_shared_ssdag
      else:
         PING('failed\n      if "bbs" in kw and (len(kw.bbs) > 2 or kw.bbs[0] is not kw.bbs[1]):')
         assert 0
         ####
      kw.timer.checkpoint('make _global_shared_ssdag')
      #

      if _global_shared_ssdag is None:
         assert 0, 'no _global_shared_ssdag??'
      if not "bbs" in kw:
         kw.bbs = _global_shared_ssdag.bbs
      assert len(_global_shared_ssdag.bbs) == len(kw.bbs)
      for a, b in zip(_global_shared_ssdag.bbs, kw.bbs):
         for aa, bb in zip(a, b):
            assert aa is bb
      PING('_global_shared_ssdag complete')

      if kw.context_structure:
         print('have context_structure')
         kw.context_structure = worms.clashgrid.ClashGrid(kw.context_structure, **kw)
      else:
         kw.context_structure = None

      kw.timer.checkpoint('construct_global_ssdag_and_run')
      log = run_all_mbblocks(criteria, **kw)
      kw.timer.checkpoint('run_all_mbblocks')

      PING('run_all_mbblocks returned')
      if kw.pbar:
         print("======================== logs ========================")
         for msg in log:
            print(msg)
   print("======================== done ========================")
   return Bunch(log=log, ssdag=_global_shared_ssdag, database=kw.database, strict__=True)

def run_all_mbblocks(
   criteria,
   precache_splices,
   merge_bblock,
   parallel,
   verbosity,
   bbs,
   pbar,
   only_merge_bblocks,
   merge_segment,
   **kw,
):
   kw = Bunch(kw)
   print('run_all_mbblocks start', flush=True)
   exe = util.InProcessExecutor()
   if parallel:
      exe = cf.ProcessPoolExecutor(max_workers=parallel)

   bbs_states = [[b._state for b in bb] for bb in bbs]

   # kw.database.bblockdb.clear_bblocks()  # remove cached BBlocks
   kw.database.bblockdb.clear()
   kw.database.splicedb.clear()
   kw.timer.checkpoint('run_all_mbblocks')

   with exe as pool:
      mseg = merge_segment
      if mseg is None:
         mseg = criteria.merge_segment(**kw)
      if mseg is None:
         mseg = 0
      merge_bblock_list = range(len(bbs[mseg]))
      if only_merge_bblocks:
         merge_bblock_list = only_merge_bblocks
      print('   mergebblist:', merge_bblock_list)
      futures = [
         pool.submit(
            run_one_mbblock,
            criteria,
            merge_bblock=i,
            parallel=0,
            verbosity=verbosity,
            bbs_states=bbs_states,
            precache_splices=precache_splices,
            pbar=pbar,
            merge_segment=merge_segment,
            **kw,
         ) for i in merge_bblock_list
      ]
      kw.timer.checkpoint('run_all_mbblocks: submit_jobs')
      log = [f"split job over merge_segment={mseg}, n = {len(futures)}"]
      # print(log[-1])

      fiter = cf.as_completed(futures)  # type: ignore

      for f in fiter:
         PING(f'{merge_bblock} f in fiter')
         log.extend(f.result())
      if pbar and log:
         log = [""] * len(futures) + log

      print(merge_bblock, 'run_all_mbblocks done', flush=True)
      kw.timer.checkpoint('run_all_mbblocks: finish_jobs')

      return log

def run_one_mbblock(
   criteria,
   bbs_states=None,
   disable_clash_check=0,
   return_raw_result=False,
   **kw,
):
   kw = Bunch(kw)
   print('================= run_one_mbblock', kw.merge_bblock, '=================')
   kw.timer.checkpoint()
   # try:
   if True:
      if bbs_states is not None:
         kw.bbs = [tuple(worms.bblock._BBlock(*s) for s in bb) for bb in bbs_states]

      ssdag, result1, log = search_all_stages(criteria, **kw)
      kw.timer.checkpoint('search_all_stages')

      if result1 is None:
         return []

      if True:
         result1 = worms.filters.prune_duplicates_on_segpos(result1)
      kw.timer.checkpoint('prune_duplicates_on_segpos')

      if disable_clash_check:
         result2 = result1
      else:
         result2 = prune_clashes(ssdag, criteria, result1, **kw)
      kw.timer.checkpoint('prune_clashes')

      result3 = check_geometry(ssdag, criteria, result2, **kw)
      kw.timer.checkpoint('check_geometry')

      log = []
      if True:  # len(result3.idx) > 0:
         msg = f'mbb{kw.merge_bblock:04} nresults after clash/geom check {len(result3.idx):,}'
         log.append("    " + msg)
         print(log[-1])

      if return_raw_result:
         return [result3]

      kw.timer.checkpoint('run_one_mbblock')
      r = worms.output.filter_and_output_results(
         criteria,
         ssdag,
         result3,
         **kw,
      )
      log += r
      kw.timer.checkpoint('filter_and_output_results')

      if not kw.pbar:
         print(f'completed: mbb{kw.merge_bblock:04}')
         sys.stdout.flush()

      return log

   # except Exception as e:
   #    print("error on mbb" + str(kw.merge_bblock))
   #    print(type(e))
   #    print(traceback.format_exc())
   #    print(e)
   #    sys.stdout.flush()
   #    return []

def search_all_stages(
   criteria,
   bbs,
   monte_carlo,
   merge_segment,
   **kw,
):
   kw = Bunch(kw)
   stages = [(criteria, bbs)]
   merge = None
   kw.timer.checkpoint()
   if hasattr(criteria, "stages"):
      stages, merge = criteria.stages(bbs=bbs, **kw)
   if len(stages) > 1:
      assert kw.merge_bblock is not None
   kw.timer.checkpoint('search_all_stages: get stages')
   assert len(monte_carlo) in (1, len(stages))
   if len(monte_carlo) != len(stages):
      monte_carlo *= len(stages)

   results = list()
   for i, stage in enumerate(stages):
      crit, stage_bbs = stage
      if callable(crit):
         crit = crit(*results[-1][:-1])  # TODO wtf is this?
      lbl = f"stage{i}"
      if kw.merge_bblock is not None:
         lbl = f'stage{i}_mbb{kw.merge_bblock:04}'
      PING('start search_single_stage')

      kw.timer.checkpoint('search_all_stages')
      single_stage_result = search_single_stage(
         crit,
         monte_carlo=monte_carlo[i],
         lbl=lbl,
         bbs=stage_bbs,
         merge_segment=merge_segment,
         **kw,
      )
      kw.timer.checkpoint('search_single_stage')

      results.append(single_stage_result)
      if not hasattr(crit, "produces_no_results") and len(results[-1][2].idx) == 0:
         print("mbb", kw.merge_bblock, "no results at stage", i)
         return None, None, None

   # todo: this whole block is very protocol-specific... needs refactoring
   if len(results) == 1:
      assert merge is None
      return results[0][1:]
   elif len(results) == 2:
      assert merge is not None
      mseg = merge_segment
      if mseg is None:
         mseg = criteria.merge_segment(**kw)
      # simple_search_dag getting not-to-simple maybe split?
      _____, ssdA, rsltA, logA = results[0]
      critB, ssdB, rsltB, logB = results[1]
      assert _global_shared_ssdag
      print('main.py:search_all_stages calling simple_search_dag', flush=True)
      ssd = simple_search_dag(
         criteria,
         only_seg=mseg,
         make_edges=False,
         source=_global_shared_ssdag,
         bbs=bbs,
         **kw,
      )
      ssdag = ssd.ssdag
      assert ssdag is not None
      print('main.py:search_all_stages calling simple_search_dag DONE', flush=True)
      ssdag.verts = ssdB.verts[:-1] + (ssdag.verts[mseg], ) + ssdA.verts[1:]

      assert len(ssdag.verts) == len(criteria.bbspec)
      rslt = merge(criteria, ssdag, ssdA, rsltA, critB, ssdB, rsltB, **kw)
      kw.timer.checkpoint('search_all_stages')
      return ssdag, rslt, logA + logB

   elif len(results) == 3:
      # hacky: assume 3stage is brigde protocol
      assert merge is not None
      _, _, _, logA = results[0]
      _, ssdB, _, logB = results[1]
      critC, ssdC, rsltC, logC = results[2]

      assert _global_shared_ssdag
      mseg = merge_segment
      if mseg is None:
         mseg = criteria.merge_segment(**kw)
      ssd = simple_search_dag(
         criteria,
         only_seg=mseg,
         make_edges=False,
         source=_global_shared_ssdag,
         bbs=bbs,
         **kw,
      )
      ssdag = ssd.ssdag
      assert ssdag is not None
      ssdag.verts = ssdC.verts[:-1] + (ssdag.verts[mseg], ) + ssdB.verts[1:]
      assert len(ssdag.verts) == len(criteria.bbspec)

      rslt = merge(criteria, critC, ssdag, ssdB, ssdC, rsltC, **kw)

      kw.timer.checkpoint('search_all_stages')
      return ssdag, rslt, logA + logB + logC

   else:

      assert 0, "unknown 3+ stage protcol"

def search_single_stage(
   criteria,
   lbl="",
   **kw,
):
   from worms.search.linear import grow_linear

   kw = Bunch(kw)
   if kw["run_cache"]:
      if os.path.exists(kw["run_cache"] + lbl + ".pickle"):
         with (open(kw["run_cache"] + lbl + ".pickle", "rb")) as inp:
            ssdag, result = pickle.load(inp)
            return criteria, ssdag, result, ["from run cache " + lbl]

   PING('call simple_search_dag')
   kw.timer.checkpoint('search_single_stage')
   ssd = simple_search_dag(
      criteria,
      source=_global_shared_ssdag,
      lbl=lbl,
      **kw,
   )
   ssdag = ssd.ssdag
   print(ssdag)
   kw.timer.checkpoint('simple_search_dag')

   PING('call grow_linear')
   result, tsearch = run_and_time(
      grow_linear,
      ssdag=ssdag,
      loss_function=criteria.jit_lossfunc(**kw),
      last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
      lbl=lbl,
      **kw,
   )
   kw.timer.checkpoint('grow_linear')
   PING('call grow_linear done')

   Nsparse = result.stats.total_samples[0]
   Nsparse_rate = int(Nsparse / tsearch)

   log = []
   if len(result.idx):
      frac_redundant = result.stats.n_redundant_results[0] / len(result.idx)
      log = [
         f"grow_linear {lbl} done, nresults {len(result.idx):,}, " +
         f"total_samples {result.stats.total_samples[0]:,}, " +
         f"samp/sec {Nsparse_rate:,}, redundant ratio {frac_redundant}"
      ]
   if log:
      print(log[-1])

   if kw["run_cache"]:
      with (open(kw["run_cache"] + lbl + ".pickle", "wb")) as out:
         pickle.dump((ssdag, result), out)
   kw.timer.checkpoint('search_single_stage')
   return criteria, ssdag, result, log
