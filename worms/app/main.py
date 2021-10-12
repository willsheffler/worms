import sys
import os
import io
import argparse
import _pickle
from copy import deepcopy
import itertools as it
from time import time
import concurrent.futures as cf
import traceback
from worms import Bunch

import pyrosetta
import blosc
from tqdm import tqdm
# from xbin import gu_xbin_indexer, numba_xbin_indexer
from worms import homog as hg

from worms.cli import build_worms_setup_from_cli_args
from worms.ssdag import simple_search_dag
from worms.search import grow_linear
from worms.util import run_and_time
from worms import util
from worms.filters.clash import prune_clashes
from worms.filters.geometry import check_geometry
from worms.bblock import _BBlock
from worms.clashgrid import ClashGrid
from worms.output import filter_and_output_results

_shared_ssdag = None

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

   pyrosetta.init("-mute all -beta -preserve_crystinfo --prevent_repacking")
   blosc.set_releasegil(True)

   criteria_list, kw = build_worms_setup_from_cli_args(argv)

   try:
      worms_main2(criteria_list, kw)
   except Exception as e:
      bbdb = kw["db"][0]
      bbdb.clear()
      raise e

   print('worms_main done, time:', time() - tstart)

def worms_main2(criteria_list, kw):
   print("worms_main2,", len(criteria_list), "criteria, args:")
   orig_output_prefix = kw["output_prefix"]
   for icrit, criteria in enumerate(criteria_list):
      if len(criteria_list) > 1:
         assert len(criteria_list) is len(kw["config_file"])
         name = os.path.basename(kw["config_file"][icrit])
         name = name.replace(".config", "")
         kw["output_prefix"] = orig_output_prefix + "_" + name
      print("================== start job", icrit, "======================")
      print("output_prefix:", kw["output_prefix"])
      print("criteria:", criteria)
      print("bbspec:", criteria.bbspec, flush=True)

      if kw["precache_splices"]:
         print("precaching splices")
         merge_bblock = kw["merge_bblock"]
         del kw["merge_bblock"]
         pbar = kw["pbar"]
         del kw["pbar"]
         kw["bbs"] = simple_search_dag(criteria, merge_bblock=None, precache_only=True, pbar=True,
                                       **kw)
         if kw["only_bblocks"]:
            assert len(kw["bbs"]) is len(kw["only_bblocks"])
            for i, bb in enumerate(kw["bbs"]):
               kw["bbs"][i] = [bb[kw["only_bblocks"][i]]]
            print("modified bblock numbers (--only_bblocks)")
            print("   ", [len(b) for b in kw["bbs"]])
         kw["merge_bblock"] = merge_bblock
         kw["pbar"] = pbar
         if kw["precache_splices_and_quit"]:
            return None

      global _shared_ssdag
      if "bbs" in kw and (len(kw["bbs"]) > 2 or kw["bbs"][0] is not kw["bbs"][1]):

         ############3

         #

         # _shared_ssdag = simple_search_dag(
         #    criteria, print_edge_summary=True, **kw
         # )

         merge_bblock = kw["merge_bblock"]
         del kw["merge_bblock"]
         _shared_ssdag = simple_search_dag(criteria, merge_bblock=0, print_edge_summary=True,
                                           **kw)
         kw["merge_bblock"] = merge_bblock
         print("memuse for global _shared_ssdag:")
         _shared_ssdag.report_memory_use()
         assert _shared_ssdag
      else:
         print(
            'failed\n      if "bbs" in kw and (len(kw["bbs"]) > 2 or kw["bbs"][0] is not kw["bbs"][1]):'
         )
         assert 0
         ####

         #

      if _shared_ssdag:
         if not "bbs" in kw:
            kw["bbs"] = _shared_ssdag.bbs
         assert len(_shared_ssdag.bbs) == len(kw["bbs"])
         for a, b in zip(_shared_ssdag.bbs, kw["bbs"]):
            for aa, bb in zip(a, b):
               assert aa is bb
         print('_shared_ssdag complete', flush=True)
      else:
         assert 0, 'no _shared_ssdag??'

      if kw["context_structure"]:
         print('have context_structure')
         kw["context_structure"] = ClashGrid(kw["context_structure"], **kw)
      else:
         kw["context_structure"] = None

      log = worms_main_each_mergebb(criteria, **kw)
      print('worms_main_each_mergebb returned', flush=True)
      if kw["pbar"]:
         print("======================== logs ========================")
         for msg in log:
            print(msg)
   print("======================== done ========================")
   return Bunch(log=log, ssdag=_shared_ssdag, database=kw['db'])

def worms_main_each_mergebb(
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
   print('worms_main_each_mergebb start', flush=True)
   exe = util.InProcessExecutor()
   if parallel:
      exe = cf.ProcessPoolExecutor(max_workers=parallel)

   bbs_states = [[b._state for b in bb] for bb in bbs]
   # kw['db'][0].clear_bblocks()  # remove cached BBlocks
   kw["db"][0].clear()
   kw["db"][1].clear()

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
            worms_main_protocol,
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
      log = [f"split job over merge_segment={mseg}, n = {len(futures)}"]
      # print(log[-1])

      fiter = cf.as_completed(futures)
      for f in fiter:
         print(merge_bblock, 'f in fiter', flush=True)
         log.extend(f.result())
      if pbar and log:
         log = [""] * len(futures) + log

      print(merge_bblock, 'worms_main_each_mergebb done', flush=True)

      return log

def worms_main_protocol(criteria, bbs_states=None, disable_clash_check=0, return_raw_result=False,
                        **kw):

   try:
      if bbs_states is not None:
         kw["bbs"] = [tuple(_BBlock(*s) for s in bb) for bb in bbs_states]

      ssdag, result1, log = search_func(criteria, **kw)
      if result1 is None:
         return []

      if disable_clash_check:
         result2 = result1
      else:
         result2 = prune_clashes(ssdag, criteria, result1, **kw)

      result3 = check_geometry(ssdag, criteria, result2, **kw)

      log = []
      if True:  # len(result3.idx) > 0:
         msg = f'mbb{kw["merge_bblock"]:04} nresults after clash/geom check {len(result3.idx):,}'
         log.append("    " + msg)
         print(log[-1])

      if not return_raw_result:
         log += filter_and_output_results(criteria, ssdag, result3, **kw)
      elif return_raw_result:
         # print('!' * 60)
         # print('returning raw result')
         # print('!' * 60)
         return [result3]
      else:
         print('logical impossibility')

      if not kw["pbar"]:
         print(f'completed: mbb{kw["merge_bblock"]:04}')
         sys.stdout.flush()

      return log

   except Exception as e:
      print("error on mbb" + str(kw["merge_bblock"]))
      print(type(e))
      print(traceback.format_exc())
      print(e)
      sys.stdout.flush()
      return []

def search_func(criteria, bbs, monte_carlo, merge_segment, **kw):

   stages = [(criteria, bbs)]
   merge = None
   if hasattr(criteria, "stages"):
      stages, merge = criteria.stages(bbs=bbs, **kw)
   if len(stages) > 1:
      assert kw["merge_bblock"] is not None

   assert len(monte_carlo) in (1, len(stages))
   if len(monte_carlo) != len(stages):
      monte_carlo *= len(stages)

   results = list()
   for i, stage in enumerate(stages):
      crit, stage_bbs = stage
      if callable(crit):
         crit = crit(*results[-1][:-1])
      lbl = f"stage{i}"
      if kw["merge_bblock"] is not None:
         lbl = f'stage{i}_mbb{kw["merge_bblock"]:04}'
      print('main.py:search_func: results.append( search_single_stage(')
      results.append(
         search_single_stage(
            crit,
            monte_carlo=monte_carlo[i],
            lbl=lbl,
            bbs=stage_bbs,
            merge_segment=merge_segment,
            **kw,
         ))
      if not hasattr(crit, "produces_no_results") and len(results[-1][2].idx) is 0:
         print("mbb", kw["merge_bblock"], "no results at stage", i)
         return None, None, None

   # todo: this whole block is very protocol-specific... needs refactoring
   if len(results) is 1:
      assert merge is None
      return results[0][1:]
   elif len(results) is 2:

      mseg = merge_segment
      if mseg is None:
         mseg = criteria.merge_segment(**kw)
      # simple_search_dag getting not-to-simple maybe split?
      _____, ssdA, rsltA, logA = results[0]
      critB, ssdB, rsltB, logB = results[1]
      assert _shared_ssdag
      print('main.py:search_func calling simple_search_dag', flush=True)
      ssdag = simple_search_dag(
         criteria,
         only_seg=mseg,
         make_edges=False,
         source=_shared_ssdag,
         bbs=bbs,
         **kw,
      )
      print('main.py:search_func calling simple_search_dag DONE', flush=True)
      ssdag.verts = ssdB.verts[:-1] + (ssdag.verts[mseg], ) + ssdA.verts[1:]

      assert len(ssdag.verts) == len(criteria.bbspec)
      rslt = merge(criteria, ssdag, ssdA, rsltA, critB, ssdB, rsltB, **kw)
      return ssdag, rslt, logA + logB

   elif len(results) is 3:
      # hacky: assume 3stage is brigde protocol

      _____, ____, _____, logA = results[0]
      _____, ssdB, _____, logB = results[1]
      critC, ssdC, rsltC, logC = results[2]

      assert _shared_ssdag
      mseg = merge_segment
      if mseg is None:
         mseg = criteria.merge_segment(**kw)
      ssdag = simple_search_dag(
         criteria,
         only_seg=mseg,
         make_edges=False,
         source=_shared_ssdag,
         bbs=bbs,
         **kw,
      )
      ssdag.verts = ssdC.verts[:-1] + (ssdag.verts[mseg], ) + ssdB.verts[1:]
      assert len(ssdag.verts) == len(criteria.bbspec)

      rslt = merge(criteria, critC, ssdag, ssdB, ssdC, rsltC, **kw)

      return ssdag, rslt, logA + logB + logC

   else:

      assert 0, "unknown 3+ stage protcol"

def search_single_stage(criteria, lbl="", **kw):

   if kw["run_cache"]:
      if os.path.exists(kw["run_cache"] + lbl + ".pickle"):
         with (open(kw["run_cache"] + lbl + ".pickle", "rb")) as inp:
            ssdag, result = _pickle.load(inp)
            return criteria, ssdag, result, ["from run cache " + lbl]

   print('main.py:search_single_stage calling simple_search_dag', flush=True)
   ssdag = simple_search_dag(criteria, source=_shared_ssdag, lbl=lbl, **kw)
   print('main.py:search_single_stage calling simple_search_dag DONE', flush=True)

   print('main.py:search_single_stage calling grow_linear', flush=True)
   result, tsearch = run_and_time(
      grow_linear,
      ssdag=ssdag,
      loss_function=criteria.jit_lossfunc(**kw),
      last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
      lbl=lbl,
      **kw,
   )
   print('main.py:search_single_stage calling grow_linear DONE', flush=True)

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
         _pickle.dump((ssdag, result), out)

   return criteria, ssdag, result, log
