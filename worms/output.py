import sys
import os
import psutil
import gc

from pympler.asizeof import asizeof

from pyrosetta import rosetta as ros

from worms import util
from worms.ssdag_pose import make_pose_crit
from worms.filters.db_filters import run_db_filters
from worms.filters.db_filters import get_affected_positions
from worms.ssdag import graph_dump_pdb

def getmem():
   mem = psutil.Process(os.getpid()).memory_info().rss / 2**20
   return f"{int(mem):5}"

def filter_and_output_results(
      criteria,
      ssdag,
      result,
      output_from_pose,
      merge_bblock,
      db,
      output_symmetric,
      output_centroid,
      output_prefix,
      max_output,
      max_score0,
      max_score0sym,
      rms_err_cut,
      no_duplicate_bases,
      output_only_AAAA,
      full_score0sym,
      output_short_fnames,
      output_only_connected,
      null_base_names,
      only_outputs,
      **kw,
):

# Takes in arrays of transforms of objects coming out of search function 
# idx: tells you where indices are 
# error: geometry error 
   sf = ros.core.scoring.ScoreFunctionFactory.create_score_function("score0")
   if hasattr(ros.core.scoring.symmetry, 'symmetrize_scorefunction'):
      sfsym = ros.core.scoring.symmetry.symmetrize_scorefunction(sf)
   else:
      sfsym = sf

   if max_score0sym == 9e9:
      # TODO: improve this logic?
      max_score0sym = 2.0 * max_score0

   mbb = ""
   if merge_bblock is not None:
      mbb = f"_mbb{merge_bblock:04d}"

   head = f"{output_prefix}{mbb}"
   if mbb and output_prefix[-1] != "/":
      head += "_"

   if not merge_bblock:
      # do this once per run, at merge_bblock == 0 (or None)
      with open(head + "__HEADER.info", "w") as info_file:
         info_file.write("close_err close_rms score0 score0sym filter zheight zradius " +
                         "radius porosity nc nc_wo_jct n_nb bases_str fname nchain chain_len " +
                         "splicepoints ibblocks ivertex")
         N = len(ssdag.verts)
         info_file.write(" seg0_pdb_0 seg0_exit")
         for i in range(1, N - 1):
            info_file.write(" seg%i_enter seg%i_pdb seg%i_exit" % (i, i, i))
         info_file.write(" seg%i_enter seg%i_pdb" % (N - 1, N - 1))
         info_file.write("\n")

   if output_from_pose:
      info_file = None
      nresults = 0
      Ntotal = min(max_output, len(result.idx))
      for iresult in range(Ntotal):

         if only_outputs and iresult not in only_outputs:
            print('output skipping', iresult)
            continue

         if False: 
            # make json files with bblocks for single result
            tmp, seenit = list(), set()
            for j in range(len(ssdag.verts)):
               v = ssdag.verts[j]
               ibb = v.ibblock[result.idx[iresult, j]]
               bb = ssdag.bbs[j][ibb]
               fname = str(bytes(bb.file), 'utf-8')
               if fname not in seenit:
                  for e in db[0]._alldb:
                     if e['file'] == fname:
                        tmp.append(e)
               seenit.add(fname)
            import json
            jsonfname = 'tmp_%i.json' % iresult
            print('output bblocks to', jsonfname)
            with open(jsonfname, 'w') as out:
               json.dump(tmp, out)

         # print(getmem(), 'MEM ================ top of loop ===============')

         if iresult % 100 == 0:
            process = psutil.Process(os.getpid())
            gc.collect()
            mem_before = process.memory_info().rss / float(2**20)
            db[0].clear()
            gc.collect()
            mem_after = process.memory_info().rss / float(2**20)
            print("clear db", mem_before, mem_after, mem_before - mem_after)

         if iresult % 10 == 0:
            process = psutil.Process(os.getpid())
            if hasattr(db[0], "_poses_cache"):
               print(
                  f"mbb{merge_bblock:04} dumping results {iresult} of {Ntotal}",
                  "pose_cache",
                  sys.getsizeof(db[0]._poses_cache),
                  len(db[0]._poses_cache),
                  f"{process.memory_info().rss / float(2**20):,}mb",
               )

         bases = ssdag.get_bases(result.idx[iresult])
         bases_str = ",".join(bases)
         if no_duplicate_bases:
            if criteria.is_cyclic:
               bases = bases[:-1]
            for null_name in null_base_names:
               while null_name in bases:
                  bases.remove(null_name)
            bases_uniq = set(bases)
            nbases = len(bases)
            if len(bases_uniq) != nbases:
               if criteria.is_cyclic:
                  bases[-1] = "(" + bases[-1] + ")"
               print("duplicate bases fail", merge_bblock, iresult, bases)
               continue

         try:
         	# makes a pose and tells you where it came from and the building blocks and sub poses 
            # print(getmem(), 'MEM make_pose_crit before')
            pose, prov = make_pose_crit(
               db[0],
               ssdag,
               criteria,
               result.idx[iresult],
               result.pos[iresult],
               only_connected=output_only_connected,
               provenance=True,
               # full_output_segs=[0],
            )
            # print(getmem(), 'MEM make_pose_crit after')
         except ValueError as e:
            print("error in make_pose_crit:")
            print(e)
            continue

         # print(getmem(), 'MEM dbfilters before')
         try:
            (
               jstr,
               jstr1,
               filt,
               grade,
               sp,
               mc,
               mcnh,
               mhc,
               nc,
               ncnh,
               nhc,
            ) = run_db_filters(db, criteria, ssdag, iresult, result.idx[iresult], pose, prov,
                               **kw) # TODO: see what filters are being run 
         except Exception as e:
            print("error in db_filters:")
            print(traceback.format_exc())
            print(e)
            continue
         # print(getmem(), 'MEM dbfilters after')

         if output_only_AAAA and grade != "AAAA":
            print(f"mbb{merge_bblock:04} {iresult:06} bad grade", grade)
            continue

         # print(getmem(), 'MEM rms before')
         rms = criteria.iface_rms(pose, prov, **kw) # Checks for how close symmetry axes are to being intersecting 
         # if rms > rms_err_cut: continue
         # print(getmem(), 'MEM rms after')

         # print(getmem(), 'MEM poses and score0 before')
         cenpose = pose.clone()
         ros.core.util.switch_to_residue_type_set(cenpose, "centroid")
         score0 = sf(cenpose) # basically a backbone clash check 
         # print(getmem(), 'MEM poses and score0 after')
         if score0 > max_score0:
            print(
               f"mbb{merge_bblock:04} {iresult:06} score0 fail",
               merge_bblock,
               iresult,
               "score0",
               score0,
               "rms",
               rms,
               "grade",
               grade,
            )
            continue

         symfilestr = None
         if hasattr(criteria, "symfile_modifiers"): 
         	# if statement for symmetry mates for unbounded stuff to adjust cell size and sym data to make and score a symmetric post, check sym files of layers and xtals for details 
            symdata, symfilestr = util.get_symdata_modified(
               criteria.symname,
               **criteria.symfile_modifiers(segpos=result.pos[iresult]),
            )
         else:
            symdata = util.get_symdata(criteria.symname)

         # print(getmem(), 'MEM poses and score0sym before')
         if symdata:
            sympose = cenpose.clone()
            # if pose.pdb_info() and pose.pdb_info().crystinfo().A() > 0:
            #     ros.protocols.cryst.MakeLatticeMover().apply(sympose)
            # else:
            ros.core.pose.symmetry.make_symmetric_pose(sympose, symdata)
            score0sym = sfsym(sympose)
            if full_score0sym: # for xtals and layers, apply symmetry to get symmetry mates and do clash check
               sym_asym_pose = sympose.clone()
               ros.core.pose.symmetry.make_asymmetric_pose(sym_asym_pose) 
               score0sym = sf(sym_asym_pose)
            # print(getmem(), 'MEM poses and score0sym after')

            if score0sym >= max_score0sym:
               print(
                  f"mbb{merge_bblock:06} {iresult:04} score0sym fail",
                  score0sym,
                  "rms",
                  rms,
                  "grade",
                  grade,
               )
               continue
         else:
            score0sym = -1

         mbbstr = "None"
         if merge_bblock is not None:
            mbbstr = f"{merge_bblock:4d}"

         # print(getmem(), 'MEM chains before')
         chains = pose.split_by_chain()
         chain_info = "%4d " % (len(list(chains)))
         chain_info += "-".join(str(len(c)) for c in chains)
         # print(getmem(), 'MEM chains after')

         # print(getmem(), 'MEM get_affected_positions before')
         mod, new, lost, junct = get_affected_positions(cenpose, prov)
         # print(getmem(), 'MEM get_affected_positions after')

         if output_short_fnames:
            fname = "%s_%04i" % (head, iresult)
         else:
            jpos = "-".join(str(x) for x in junct)
            fname = "%s_%04i_%s_%s_%s" % (head, iresult, jpos, jstr[:200], grade)

         # report bblock ids, taking into account merge_bblock shenani
         ibblock_list = [str(v.ibblock[i]) for i, v in zip(result.idx[iresult], ssdag.verts)]
         mseg = kw["merge_segment"]
         mseg = criteria.merge_segment(**kw) if mseg is None else mseg
         mseg = mseg or 0  # 0 if None
         # print("!!!!!!!", merge_bblock, "mseg", mseg, ibblock_list)
         ibblock_list[mseg] = str(merge_bblock)

         if not info_file:
            d = os.path.dirname(output_prefix)
            if d != "" and not os.path.exists(d):
               os.makedirs(d)
            info_file = open(f"{output_prefix}{mbb}.info", "w")
         info_file.write(
            "%5.2f %5.2f %7.2f %7.2f %-8s %5.1f %5.1f %5.1f %5.3f %4d %4d %4d %s %-80s %s  %s %s %s %s\n"
            % (
               result.err[iresult],
               rms,
               score0,
               score0sym,
               grade,
               result.zheight[iresult],
               result.zradius[iresult],
               result.radius[iresult],
               result.porosity[iresult],
               mc,
               mcnh,
               mhc,
               bases_str,
               fname,
               chain_info,
               "-".join([str(x) for x in sp]),
               "-".join(ibblock_list),
               "-".join(str(x) for x in result.idx[iresult]),
               jstr1,
            ))
         info_file.flush()

         # print(getmem(), 'MEM dump pdb before')
         if symdata and output_symmetric:
            sympose.dump_pdb(fname + "_sym.pdb")
         if output_centroid:
            pose = cenpose
         print("solution", fname)
         pose.dump_pdb(fname + "_asym.pdb")
         if symfilestr is not None:
            with open(fname + ".sym", "w") as out:
               out.write(symfilestr)
         nresults += 1
         commas = lambda l: ",".join(str(_) for _ in l)
         with open(fname + "_asym.pdb", "a") as out:
            for ip, p in enumerate(prov):
               lb, ub, psrc, lbsrc, ubsrc = p
               out.write(f"Segment: {ip:2} resis {lb:4}-{ub:4} come from resis " +
                         f"{lbsrc}-{ubsrc} of {psrc.pdb_info().name()}\n")
            nchain = pose.num_chains()
            out.write("Bases: " + bases_str + "\n")
            out.write("Modified positions: " + commas(mod) + "\n")
            out.write("New contact positions: " + commas(new) + "\n")
            out.write("Lost contact positions: " + commas(lost) + "\n")
            out.write("Junction residues: " + commas(junct) + "\n")
            out.write("Length of asymetric unit: " + str(len(pose.residues)) + "\n")
            out.write("Number of chains in ASU: " + str(nchain) + "\n")
            out.write("Closure error: " + str(rms) + "\n")
         #

         print(getmem(), 'MEM dump pdb after')

      if info_file is not None:
         info_file.close()

   else:
   	# dump raw structure data from different positions in ssdag array 
   	# result object: indices and positions and array of errors from search 
      nresults = 0
      for iresult in range(min(max_output, len(result.idx))):
         fname = "%s_%04i" % (head, iresult)
         print(result.err[iresult], fname)
         graph_dump_pdb(
            fname + ".pdb",
            ssdag,
            result.idx[iresult],
            result.pos[iresult],
            join="bb", #if join is false, it'll give you all separate chains from all segments splicing together
            trim=True, # If trim is false it'll give you the whole building blocks that all overlapping 
         )
         nresults += 1

   if nresults:
      return ["nresults output" + str(nresults)]
   else:
      return []
