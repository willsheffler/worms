from posix import _exit
import sys, collections, os, psutil, gc, json, traceback, copy

# from pympler.asizeof import asizeof

import worms
from worms import util, Bunch, PING
from worms.ssdag import make_pose_crit
from worms.output.dumppdb import graph_dump_pdb
import willutil as wu

from deferred_import import deferred_import

db_filters = deferred_import('worms.filters.db_filters')
ros = deferred_import('worms.rosetta_init')
util = deferred_import('worms.util.rosetta_utils')

def getmem():
   mem = psutil.Process(os.getpid()).memory_info().rss / 2**20
   return f"{int(mem):5}"

def filter_and_output_results(
   criteria,
   ssdag,
   result,
   output_from_pose,
   merge_bblock,
   database,
   output_symmetric,
   output_centroid,
   output_prefix,
   max_output,
   max_score0,
   max_score0sym,
   # rms_err_cut,
   no_duplicate_bases,
   output_only_AAAA,
   full_score0sym,
   output_short_fnames,
   output_only_connected,
   null_base_names,
   only_outputs,
   debug=False,
   use_simple_pose_construction=False,
   align_with_chainbreak=False,
   output_extra_alignments=False,
   **kw,
):
   kw = Bunch(kw)
   if not use_simple_pose_construction and kw.repeat_add_to_output != [0]:
      print('force use_simple_pose_construction', kw.repeat_add_to_output)
      assert 0
      use_simple_pose_construction = True

   print_pings = debug
   files_output = list()

   numfail = Bunch(xalign=0, crystinfo=0, cell_to_small=0, cell_to_big=0, duplicate_bases=0,
                   make_pose_crit=0, redundant=0, only_AAAA=0, score0=0, score0sym=0, symops=0,
                   only_outputs=0, __strict=True)

   PING('mbb%i' % merge_bblock, print_pings)

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
      PING('mbb%i' % merge_bblock, print_pings)
      # do this once per run, at merge_bblock == 0 (or None)
      outdir = os.path.dirname(head)
      with open(head + "__HEADER.info", "w") as info_file:
         PING('mbb%i' % merge_bblock, print_pings)
         info_file.write("close_err close_rms score0 score0sym filter zheight zradius " +
                         "radius porosity nc nc_wo_jct n_nb bases_str fname nchain chain_len " +
                         "splicepoints ibblocks ivertex")
         N = len(ssdag.verts)
         info_file.write(" seg0_pdb_0 seg0_exit")
         for i in range(1, N - 1):
            info_file.write(" seg%i_enter seg%i_pdb seg%i_exit" % (i, i, i))
         info_file.write(" seg%i_enter seg%i_pdb" % (N - 1, N - 1))
         info_file.write("\n")

   nresults, npdbs_dumped = 0, 0

   result_json = list()

   if not output_from_pose:

      PING('mbb%i no pose output' % merge_bblock, print_pings)

      for iresult in range(min(max_output, len(result.idx))):

         PING('mbb%i' % merge_bblock, print_pings)

         segpos = result.pos[iresult]
         xalign = criteria.alignment(segpos)
         assert xalign is not None
         if xalign is None:
            numfail.xalign += 1
            continue

         cystinfo = None
         if hasattr(criteria, "crystinfo"):
            crystinfo = criteria.crystinfo(segpos=result.pos[iresult])
            if crystinfo is not None:
               if crystinfo[0] < kw.xtal_min_cell_size:
                  numfail.cell_to_small += 1
                  continue
               if crystinfo[0] > kw.xtal_max_cell_size:
                  numfail.cell_to_big += 1
                  continue

         fname = "%s_%04i" % (head, iresult)
         # print('align_ax1', xalign @ segpos[0, :, 2])
         # print('align_ax2', xalign @ segpos[-1, :, 2])
         # print(fname)
         # print(result.err[iresult], fname)
         # assert not os.path.exists(fname + '.pdb')
         graph_dump_pdb(
            fname + ".pdb",
            ssdag,
            result.idx[iresult],
            result.pos[iresult],
            join="splice",
            trim=True,
            xalign=xalign,
            crystinfo=crystinfo,
            database=database,
            **kw,
         )
         npdbs_dumped += 1
         nresults += 1
         files_output.append(fname + '.pdb')

         result_json.append(
            make_json_for_result(
               ssdag,
               database,
               merge_bblock,
               result,
               iresult,
               print_pings,
               output_prefix,
               **kw,
            ))
         # assert 0

   else:
      PING('mbb%i pose output' % merge_bblock, print_pings)

      info_file = None
      Ntotal = min(max_output, len(result.idx))
      _stuff = list(range(Ntotal))
      seenpose = collections.defaultdict(lambda: list())

      for iresult in _stuff:

         # print('!' * 80)
         # xrand = wu.hrand(1, 1, 0.05, seed=7)
         # result.pos[iresult][1] = xrand @ result.pos[iresult][1]
         # result.pos[iresult][2] = xrand @ result.pos[iresult][2]
         # print('!' * 80)

         # bblock = ssdag.bblocks[iseg][ssdag.verts[iseg].ibblock[idx[iseg]]]
         for iseg in range(len(result.idx[iresult])):
            print(iseg, 'bblock')
            ivert = result.idx[iresult][iseg]
            ibblock = ssdag.verts[iseg].ibblock[ivert]
            segbbs = ssdag.bblocks[iseg]
            bblock = segbbs[ibblock]
            print(bblock.pdbfile)
            print()

         for foo in kw.repeat_add_to_output:
            assert 0 <= foo <= 100
         # print(kw.repeat_add_to_segment), len(result.idx), len(result.idx[0])
         assert 0 <= kw.repeat_add_to_segment < len(result.idx[0])

         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         for iextend in kw.repeat_add_to_output:
            extensions = {}
            if iextend > 0:
               extensions = {kw.repeat_add_to_segment: iextend}

            PING('mbb%i' % merge_bblock, print_pings)

            if only_outputs and iresult not in only_outputs:
               print('output skipping', iresult)
               numfail.only_outputs += 1
               # print(iresult, only_outputs)
               # print(numfail)
               # assert 0
               continue

            crystinfo = None
            if hasattr(criteria, "crystinfo"):
               crystinfo = criteria.crystinfo(segpos=result.pos[iresult])
               if crystinfo:
                  assert not extensions
                  if crystinfo[0] < kw.xtal_min_cell_size:
                     numfail.cell_to_small += 1
                     continue
                  if crystinfo[0] > kw.xtal_max_cell_size:
                     numfail.cell_to_big += 1
                     continue
                     # locally trying i432 cagextal -- issue with symops return none on xalign fail
                     # digs trying p432 again

            # print(getmem(), 'MEM ================ top of loop ===============')

            if iresult % 100 == 0:
               PING('mbb%i' % merge_bblock, print_pings)
               process = psutil.Process(os.getpid())
               gc.collect()
               mem_before = process.memory_info().rss / float(2**20)
               database.bblockdb.clear()
               gc.collect()
               mem_after = process.memory_info().rss / float(2**20)
               print("clear database", mem_before, mem_after, mem_before - mem_after)

            # if iresult % 10 == 0:
            if iresult % 1 == 0:
               PING('mbb%i' % merge_bblock, print_pings)
               process = psutil.Process(os.getpid())
               if hasattr(database.bblockdb, "_poses_cache"):
                  print(merge_bblock, iresult, Ntotal)
                  print(
                     f"mbb{merge_bblock:04} checking results {iresult} of {Ntotal}",
                     "pose_cache",
                     sys.getsizeof(database.bblockdb._poses_cache),
                     len(database.bblockdb._poses_cache),
                     f"{process.memory_info().rss / float(2**20):,}mb",
                  )

            PING('mbb%i' % merge_bblock, print_pings)
            bases = ssdag.get_bases(result.idx[iresult])
            bases_str = ",".join(bases)
            if no_duplicate_bases:
               PING('mbb%i' % merge_bblock, print_pings)
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
                  numfail.duplicate_bases += 1
                  continue

            extpos = result.pos[iresult]
            if use_simple_pose_construction:
               # if True:
               print('output.py: use_simple_pose_construction')
               sinfo = ssdag.get_structure_info(result.idx[0])
               # xalign = criteria.alignment(result.pos[iresult])
               xalign = criteria.alignment(result.pos[iresult], alignto='mid')

               if isinstance(criteria, worms.criteria.AxesIntersect):
                  xalign, extpos = worms.extension.modify_xalign_cage_by_extension(
                     ssdag,
                     result.idx[iresult],
                     result.pos[iresult],
                     xalign,
                     sinfo.bblocks,
                     database.bblockdb,
                     extensions=extensions,
                     database=database,
                     **kw,
                  )
               elif criteria.is_cyclic:
                  xalign, extpos = worms.extension.modify_xalign_cyclic_by_extension(
                     ssdag,
                     result.idx[iresult],
                     result.pos[iresult],
                     xalign,
                     sinfo.bblocks,
                     database.bblockdb,
                     extensions=extensions,
                     database=database,
                     **kw,
                  )

               elif isinstance(criteria, worms.criteria.AxisAngle):
                  assert not extensions
                  extpos = retult.pos[iresult]
               else:
                  assert not extensions
                  extpos = retult.pos[iresult]

               pose, prov = worms.ssdag.make_pose_simple(
                  # pose2, prov2 = worms.ssdag.make_pose_simple(
                  ssdag,
                  result.idx[iresult],
                  xalign @ extpos,
                  extensions=extensions,
                  database=database,
                  only_spliced_regions=True,
                  is_cyclic=criteria.is_cyclic,
                  **kw,
               )

               # pose.dump_pdb(f'test{i}.pdb')

               # assert 0.,mhdtsr

            else:
               print('output.py: old pose construction')
               # if True:
               try:
                  # print(getmem(), 'MEM make_pose_crit before')
                  PING('mbb%i' % merge_bblock, print_pings)

                  pose, prov = make_pose_crit(
                     database.bblockdb,
                     ssdag,
                     criteria,
                     result.idx[iresult],
                     result.pos[iresult],
                     only_connected=output_only_connected,
                     provenance=True,
                     extensions=extensions,
                     **kw,
                     # full_output_segs=[0],
                  )

                  # print(getmem(), 'MEM make_pose_crit after')
               except ValueError as e:
                  if not kw.ignore_recoverable_errors:
                     raise e
                  print("error in make_pose_crit:")
                  print(e)
                  numfail.make_pose_crit += 1
                  continue

            # pose.dump_pdb('pose1.pdb')
            # pose2.dump_pdb('pose2.pdb')
            # for i in range(len(prov)):
            #    assert prov[i][0] == prov2[i][0]
            #    assert prov[i][1] == prov2[i][1]
            #    assert prov[i][3] == prov2[i][3]
            #    assert prov[i][4] == prov2[i][4]
            #    prov[i][2].dump_pdb(f'prov1_{i}.pdb')
            #    prov2[i][2].dump_pdb(f'prov2_{i}.pdb')

            # for p in prov:
            #   print(p)
            # assert 0
            # (1  , 109, <pyrosetta.rosetta.core.pose.Pose object at 0x7fe924d1a1b0>, 1  , 109)
            # (110, 357, <pyrosetta.rosetta.core.pose.Pose object at 0x7fe924d1a230>, 12 , 259)
            # (358, 453, <pyrosetta.rosetta.core.pose.Pose object at 0x7fe924d94fb0>, 122, 217)

            redundant = False
            for seen in seenpose[pose.size()]:
               PING('mbb%i' % merge_bblock, print_pings)
               rmsd = ros.core.scoring.CA_rmsd(seen, pose, 1, 0)  # whole pose
               if rmsd < 1.0:
                  # print('!' * 100)
                  print(f'    RMSD {iresult:04} {rmsd}')
                  # print('!' * 100
                  print('SKIPPING REDUNDANT OUTPUT')
                  redundant = True
            if redundant:
               numfail.redundant += 1
               continue
            seenpose[pose.size()].append(pose)

            # print(getmem(), 'MEM dbfilters before')
            # try:
            if True:
               PING('mbb%i' % merge_bblock, print_pings)
               # gross....
               if prov:
                  (jstr, jstr1, _, grade, sp, mc, mcnh, mhc, _, _, _) = db_filters.run_db_filters(
                     database, criteria, ssdag, iresult, result.idx[iresult], pose, prov,
                     extensions=extensions, **kw)
            # except Exception as e:
            #    print("error in db_filters:")
            #    print(traceback.format_exc())
            #    print(e)
            #    continue
            # print(getmem(), 'MEM dbfilters after')

            if output_only_AAAA and grade != "AAAA":
               print(f"mbb{merge_bblock:04} {iresult:06} bad grade", grade)
               numfail.only_AAAA += 1
               continue

            # print(getmem(), 'MEM rms before')
            PING('mbb%i' % merge_bblock, print_pings)
            rms = criteria.iface_rms(pose, prov, **kw)
            # if rms > rms_err_cut: continue
            # print(getmem(), 'MEM rms after')

            cenpose = pose.clone()
            ros.core.util.switch_to_residue_type_set(cenpose, "centroid")
            score0 = sf(cenpose)
            PING('mbb%i' % merge_bblock, print_pings)
            if score0 > max_score0:
               print(f"mbb{merge_bblock:04} {iresult:06} score0 fail", merge_bblock, iresult,
                     "score0", score0, "rms", rms, "grade", grade)
               numfail.score0 += 1
               continue

            PING('mbb%i' % merge_bblock, print_pings)

            symops = criteria.symops(segpos=result.pos[iresult])
            if symops == list():
               numfail.symops += 1
               continue

            symfilestr = None
            score0sym = -1
            symdata = None

            f_symmetrize = None
            if symops is not None:

               # fn = f"RAW_{merge_bblock:04}_{iresult:04}_"
               # pose.dump_pdb(fn + '0.pdb')
               # ros.core.pose.remove_lower_terminus_type_from_pose_residue(sympose, 1)
               # ros.core.pose.remove_upper_terminus_type_from_pose_residue(sympose, len(sympose))
               # print('!'*60)
               # print('cell spacing', crystinfo[0])
               # print('!'*60)
               def fsym(cenpose):
                  sympose = cenpose.clone()
                  orig = sympose.clone()
                  for i, op in enumerate(symops):
                     ptmp = orig.clone()
                     util.xform_pose(op, ptmp)
                     # ptmp.dump_pdb('ptmp%i.pdb'%i)
                     ros.core.pose.append_pose_to_pose(sympose, ptmp)
                     # sympose.dump_pdb('sympose%i.pdb'%i)
                  return sympose

               f_symmetrize = fsym

               # if score0sym < max_score0sym:
               #    sympose.dump_pdb(f'symops_{merge_bblock}_{iresult}.pdb')

            else:
               usecryst = pose.pdb_info() is not None and pose.pdb_info().crystinfo().A() > 0
               usecryst &= crystinfo is not None

               PING('mbb%i' % merge_bblock, print_pings)

               # usecryst = False  # MakeLatticeMover hangs sometimes
               if usecryst:
                  print('---------------- using MakeLatticeMover -------------------')
                  print('cell size', crystinfo[0], pose.pdb_info().crystinfo().A())

                  def fsym(cenpose):
                     sympose = cenpos.clone()
                     ros.protocols.cryst.MakeLatticeMover().apply(sympose)
                     return sympose

                  f_symmetrize = fsym
               else:
                  if hasattr(criteria, "symfile_modifiers"):
                     PING('mbb%i' % merge_bblock, print_pings)
                     symdata, symfilestr = util.get_symdata_modified(
                        criteria.symname,
                        **criteria.symfile_modifiers(segpos=result.pos[iresult]),
                     )
                  else:
                     PING('mbb%i' % merge_bblock, print_pings)
                     symdata = util.get_symdata(criteria.symname)

                  def fsym(cenpose):
                     sympose = cenpose.clone()
                     ros.core.pose.symmetry.make_symmetric_pose(sympose, symdata)
                     return sympose

                  f_symmetrize = fsym

            assert f_symmetrize
            sympose = f_symmetrize(cenpose)
            score0sym = sf(sympose)

            # pose.dump_pdb('test.pdb')
            # sympose.dump_pdb('testsym.pdb')
            # assert 0

            if full_score0sym and not usecryst:
               PING('full_score0sym mbb%i' % merge_bblock, print_pings)
               sym_asym_pose = sympose.clone()
               ros.core.pose.symmetry.make_asymmetric_pose(sym_asym_pose)
               score0sym = sf(sym_asym_pose)
               if score0sym > max_score0sym:
                  print('!!!!!!!!!!!!!!!!!!!!!!!!!!', score0sym)
                  sym_asym_pose.dump_pdb('full_score0_sym_fail.pdb')
                  assert 0
            # print(getmem(), 'MEM poses and score0sym after')

            if score0sym >= max_score0sym:
               print(f"mbb{merge_bblock:06} {iresult:04} score0sym fail", score0sym, "rms", rms,
                     "grade", grade)
               numfail.score0sym += 1
               continue

            # mbbstr = "None"
            # if merge_bblock is not None:
            # mbbstr = f"{merge_bblock:4d}"

            # print(getmem(), 'MEM chains before')
            chains = pose.split_by_chain()
            chain_info = "%4d " % (len(list(chains)))
            chain_info += "-".join(str(len(c)) for c in chains)
            # print(getmem(), 'MEM chains after')

            PING('mbb%i' % merge_bblock, print_pings)

            # print(getmem(), 'MEM get_affected_positions before')
            mod, new, lost, junct = db_filters.get_affected_positions(cenpose, prov)
            # print(getmem(), 'MEM get_affected_positions after')

            if output_short_fnames:
               fname = "%s_%04i" % (head, iresult)
            else:
               jpos = "-".join(str(x) for x in junct)
               fname = "%s_%04i_%s_%s_%s" % (head, iresult, jpos, jstr[:200], grade)
            fname += '_x%i' % iextend

            # report bblock ids, taking into account merge_bblock shenani
            ibblock_list = [str(v.ibblock[i]) for i, v in zip(result.idx[iresult], ssdag.verts)]
            mseg = kw["merge_segment"]
            mseg = criteria.merge_segment(**kw) if mseg is None else mseg
            mseg = mseg or 0  # 0 if None
            # print("!!!!!!!", merge_bblock, "mseg", mseg, ibblock_list)
            ibblock_list[mseg] = str(merge_bblock)

            if not info_file:
               PING('mbb%i' % merge_bblock, print_pings)
               d = os.path.dirname(output_prefix)
               if d != "" and not os.path.exists(d):
                  os.makedirs(d)
               info_file = open(f"{output_prefix}{mbb}.info", "w")
            info_file.write(
               "%5.2f %5.2f %7.2f %7.2f %-8s %5.1f %5.1f %5.1f %5.3f %4d %4d %4d %s %-80s %s  %s %s %s %s\n"
               % (result.err[iresult], rms, score0, score0sym, grade, result.zheight[iresult],
                  result.zradius[iresult], result.radius[iresult], result.porosity[iresult], mc,
                  mcnh, mhc, bases_str, fname, chain_info, "-".join([str(x) for x in sp]),
                  "-".join(ibblock_list), "-".join(str(x) for x in result.idx[iresult]), jstr1))
            info_file.flush()

            print('-' * 36, 'output', '-' * 36)
            print(fname, flush=True)
            print('-' * 80)
            if output_centroid:
               pose = cenpose

            pose.dump_pdb(fname + "_asym.pdb")
            files_output.append(fname + '_asym.pdb')
            if symdata and output_symmetric:
               sympose.dump_pdb(fname + "_sym.pdb")
               files_output.append(fname + '_sym.pdb')

            if output_extra_alignments:
               alignbeg = criteria.alignment(extpos, alignto='beg')
               alignend = criteria.alignment(extpos, alignto='end')
               alnpose = pose.clone()
               alnpos2 = pose.clone()
               util.xform_pose(alignbeg @ xalign, alnpose)
               util.xform_pose(alignend @ xalign, alnpos2)

               rmid = int(sum(prov[1][:2]) / 2)
               nolap = 25 + (prov[1][1] - prov[1][0]) // 6
               ub = rmid + nolap
               lb = rmid - nolap
               alnpose = worms.util.rosetta_utils.subpose(alnpose, 1, ub)
               alnpos2 = worms.util.rosetta_utils.subpose(alnpos2, lb, -1)
               ros.core.pose.append_pose_to_pose(alnpose, alnpos2)

               alnpose.dump_pdb(fname + "_aln_asym.pdb")
               # alnpos2.dump_pdb(fname + "_alnend_asym.pdb")
               files_output.append(fname + '_aln_asym.pdb')
               # files_output.append(fname + '_alnend_asym.pdb')
               # sympose2 = f_symmetrize(pose2)
               # sympose3 = f_symmetrize(alnpos2)
               alnsympose = f_symmetrize(alnpose)
               # sf(sympose2)
               # sf(sympose3)
               sf(alnsympose)  # must score before output
               # npdbs_dumped += 2
               npdbs_dumped += 1
               if symdata and output_symmetric:
                  # sympose2.dump_pdb(fname + "_alnbeg_sym.pdb")
                  # sympose3.dump_pdb(fname + "_alnend_sym.pdb")
                  alnsympose.dump_pdb(fname + "_aln_sym.pdb")
                  # files_output.append(fname + '_alnbeg_sym.pdb')
                  # files_output.append(fname + '_alnend_sym.pdb')
                  files_output.append(fname + '_aln_sym.pdb')
                  npdbs_dumped += 1

            if symfilestr is not None:
               with open(fname + ".sym", "w") as out:
                  out.write(symfilestr)
            nresults += 1

            result_json.append(
               make_json_for_result(
                  ssdag,
                  database,
                  merge_bblock,
                  result,
                  iresult,
                  print_pings,
                  output_prefix,
               ))

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # assert 0

            PING('mbb%i' % merge_bblock, print_pings)

            commas = lambda l: ",".join(str(_) for _ in l)

            with open(fname + "_asym.pdb", "a") as out:
               for ip, p in enumerate(prov):
                  lb, ub, psrc, lbsrc, ubsrc = p
                  pdbname = 'UNKNOWN_PDB'
                  if psrc.pdb_info():
                     pdbname = psrc.pdb_info().name()
                  out.write(f"Segment: {ip:2} resis {lb:4}-{ub:4} come from resis " +
                            f"{lbsrc}-{ubsrc} of {pdbname}\n")
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

      PING('mbb%i' % merge_bblock, print_pings)

   PING('mbb%i' % merge_bblock, print_pings)

   numfail.nsuccess = nresults

   print(f'{" filter_and_output_results stats ":$^80}')
   for k, v in numfail.items():
      print('   ', k, v)
   print('$' * 80, flush=True)

   if nresults:
      if kw.save_minimal_replicate_database:
         # raise NotImplementedError
         print('save_minimal_replicate_database nresults', len(result_json))

         arcfile = f'{head}_minimal_replicate_database.txz'
         worms.database.merge.merge_json_databases(
            result_json,
            dump_archive=arcfile,
            overwrite=True,
            pdb_contents=database.bblockdb.pdb_contents,
         )

      return Bunch(
         log=["nresults output: " + str(nresults), 'npdbs_dumped: ' + str(npdbs_dumped)],
         files=files_output, strict__=True)
   else:
      return Bunch(log=[], files=[], strict__=True)

def make_json_for_result(
   ssdag,
   database,
   merge_bblock,
   result,
   iresult,
   print_pings,
   output_prefix,
   **kw,
):

   if merge_bblock is None: merge_bblock = -1
   PING('mbb%i' % merge_bblock, print_pings)
   # make json files with bblocks for single result
   newdb = list()
   # detail = Bunch(bblock=list(), ires=list(), isite=list(), ichain=list(), direction=list())
   for iseg in range(len(ssdag.verts)):
      v = ssdag.verts[iseg]
      ivert = result.idx[iresult, iseg]
      ibb = v.ibblock[ivert]
      bb = ssdag.bbs[iseg][ibb]
      fname = str(bytes(bb.file), 'utf-8')
      if fname.startswith(r'pdbs\\\\'):
         fname = fname[4:].replace('\\\\', '/')
      dbentry = copy.deepcopy(database.bblockdb._dictdb[fname])

      conn = dbentry['connections']
      conn.clear()

      # detail.bblock.append(fname)
      # detail.ires.append(v.ires[ivert].tolist())
      # detail.isite.append(v.isite[ivert].tolist())
      # detail.ichain.append(v.ichain[ivert].tolist())
      # detail.direction.append("NC_"[ssdag.verts[iseg].dirn[0]] + "NC_"[ssdag.verts[iseg].dirn[1]])
      # assert fname not in seenit
      # for dbentry in database.bblockdb._alldb:
      #    if dbentry['file'] == fname:
      #       newdb.append(copy.deepcopy(dbentry))
      # seenit.add(fname)
      dirn = "NC_"[v.dirn[0]] + "NC_"[v.dirn[1]]
      if dirn[0] != '_':
         conn.append(
            dict(
               chain=int(v.ichain[ivert][0]),
               direction=dirn[0],
               residues=[int(v.ires[ivert][0])],
            ))
      if dirn[1] != '_':
         conn.append(
            dict(
               chain=int(v.ichain[ivert][1]),
               direction=dirn[1],
               residues=[int(v.ires[ivert][1])],
            ))
      dbentry['connections'].sort(key=lambda x: x['chain'])
      newdb.append(dbentry)

   # for iseg, dbentry in enumerate(newdb):

   # assert 0
   # for dbentry in newdb:
   #    ires = list()
   #    isite = list()
   #    for i in range(len(detail['ires'])):
   #       if dbentry['file'] == detail['bblock'][i]:
   #          ires.append(detail['ires'][i])
   #          isite.append(detail['isite'][i])
   #    for ic in isite:
   #       if ic[0] is not -1: dbentry['connections'][ic[0]]['residues'].clear()
   #       if ic[1] is not -1: dbentry['connections'][ic[1]]['residues'].clear()
   #    for ir, ic in zip(ires, isite):
   #       if ic[0] is not -1: dbentry['connections'][ic[0]]['residues'].append(ir[0])
   #       if ic[1] is not -1: dbentry['connections'][ic[1]]['residues'].append(ir[1])

   # newdb = newdb.copy()

   jsonfname = output_prefix + '_replicate_result__mbb%04i_%04i.json' % (merge_bblock, iresult)
   print('output bblocks to', jsonfname)
   if os.path.exists(jsonfname):
      print('warning: overwriting file', jsonfname)
   with open(jsonfname, 'w') as out:
      json.dump(newdb, out, indent=4)
      out.write('\n')
   return jsonfname