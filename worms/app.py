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

import gc
import psutil
from pympler.asizeof import asizeof

from tqdm import tqdm
from xbin import gu_xbin_indexer, numba_xbin_indexer
import homog as hg

from worms.criteria import *
from worms.criteria.bridge import merge_results_bridge
from worms.database import CachingBBlockDB, CachingSpliceDB
from worms.database import NoCacheBBlockDB, NoCacheSpliceDB
from worms.ssdag import simple_search_dag, graph_dump_pdb
from worms.search import grow_linear, ResultJIT, subset_result
from worms.ssdag_pose import make_pose_crit
from worms.util import run_and_time
from worms import util
from worms.filters.clash import prune_clashes
from worms.filters.geometry import check_geometry
from worms.filters.db_filters import run_db_filters
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.criteria.hash_util import _get_hash_val
from worms.filters.db_filters import get_affected_positions
from worms.bblock import _BBlock
from worms.clashgrid import ClashGrid

import pyrosetta
from pyrosetta import rosetta as ros
import blosc


def parse_args(argv):
    args = util.get_cli_args(
        argv=argv,
        geometry=[''],
        bbconn=[''],
        config_file=[''],
        nbblocks=64,
        use_saved_bblocks=0,
        monte_carlo=[0.0],
        parallel=1,
        verbosity=2,
        precache_splices=1,
        precache_splices_and_quit=0,
        pbar=0,
        pbar_interval=10.0,
        #
        context_structure='',
        #
        cachedirs=[''],
        disable_cache=0,
        dbfiles=[''],
        load_poses=0,
        read_new_pdbs=0,
        run_cache='',
        merge_bblock=-1,
        no_duplicate_bases=1,
        shuffle_bblocks=1,
        only_merge_bblocks=[-1],
        only_bblocks=[-1],
        merge_segment=-1,
        min_seg_len=15,

        # splice stuff
        splice_rms_range=4,
        splice_max_rms=0.7,
        splice_clash_d2=3.5**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_clash_contact_range=40,
        splice_clash_contact_by_helix=1,
        splice_ncontact_cut=38,
        splice_ncontact_no_helix_cut=6,
        splice_nhelix_contacted_cut=3,
        splice_max_chain_length=450,
        #
        tolerance=1.0,
        lever=25.0,
        min_radius=0.0,
        hash_cart_resl=1.0,
        hash_ori_resl=5.0,
        merged_err_cut=999.0,
        rms_err_cut=3.0,
        ca_clash_dis=3.0,
        disable_clash_check=0,
        #
        max_linear=1000000,
        max_merge=100000,
        max_clash_check=10000,
        max_output=1000,
        max_score0=9e9,
        #
        output_from_pose=1,
        output_symmetric=1,
        output_prefix='./',
        output_centroid=0,
        output_only_AAAA=0,
        #
        cache_sync=0.003,
        #
        postfilt_splice_max_rms=0.7,
        postfilt_splice_rms_length=9,
        postfilt_splice_ncontact_cut=40,
        postfilt_splice_ncontact_no_helix_cut=2,
        postfilt_splice_nhelix_contacted_cut=3,

    )
    if args.config_file == ['']:
        args.config_file = []
    if not args.config_file:
        crit = eval(''.join(args.geometry))
        bb = args.bbconn[1::2]
        nc = args.bbconn[0::2]
        crit.bbspec = list(list(x) for x in zip(bb, nc))
        assert len(nc) == len(bb)
        assert crit.from_seg < len(bb)
        assert crit.to_seg < len(bb)
        if isinstance(crit, Cyclic) and crit.origin_seg is not None:
            assert crit.origin_seg < len(bb)
        crit = [crit]
    else:
        crit = []
        for cfile in args.config_file:
            with open(cfile) as inp:
                lines = inp.readlines()
                assert len(lines) is 2

                def orient(a, b):
                    return (a or '_') + (b or '_')

                bbnc = eval(lines[0])
                bb = [x[0] for x in bbnc]
                nc = [x[1] for x in bbnc]

                crit0 = eval(lines[1])
                crit0.bbspec = list(list(x) for x in zip(bb, nc))
                assert len(nc) == len(bb)
                assert crit0.from_seg < len(bb)
                assert crit0.to_seg < len(bb)
                if isinstance(crit0, Cyclic) and crit0.origin_seg is not None:
                    assert crit0.origin_seg < len(bb)
                crit.append(crit0)

    # oh god... fix these huge assumptions about Criteria
    for c in crit:
        # c.tolerance = args.tolerance
        c.lever = args.lever
        c.rot_tol = c.tolerance / args.lever

    if args.max_score0 > 9e8:
        args.max_score0 = 2.0 * len(crit[0].bbspec)

    if args.merge_bblock < 0: args.merge_bblock = None
    if args.only_merge_bblocks == [-1]:
        args.only_merge_bblocks = []
    if args.only_bblocks == [-1]:
        args.only_bblocks = []
    if args.merge_segment == -1:
        args.merge_segment = None

    kw = vars(args)
    if args.disable_cache:
        kw['db'] = NoCacheBBlockDB(**kw), NoCacheSpliceDB(**kw)
    else:
        kw['db'] = CachingBBlockDB(**kw), CachingSpliceDB(**kw)

    return crit, kw


_shared_ssdag = None


def worms_main(argv):

    # read inputs
    criteria_list, kw = parse_args(argv)

    try:
        worms_main2(criteria_list, kw)
    except Exception as e:
        bbdb = kw['db'][0]
        bbdb.clear()
        raise e


def worms_main2(criteria_list, kw):

    print('worms_main,', len(criteria_list), 'criteria, args:')
    for k, v in kw.items():
        print('   ', k, v)
    pyrosetta.init('-mute all -beta -preserve_crystinfo --prevent_repacking')
    blosc.set_releasegil(True)

    if kw['context_structure']:
        kw['context_structure'] = ClashGrid(kw['context_structure'], **kw)
    else:
        kw['context_structure'] = None

    orig_output_prefix = kw['output_prefix']

    for icrit, criteria in enumerate(criteria_list):
        if len(criteria_list) > 1:
            assert len(criteria_list) is len(kw['config_file'])
            name = os.path.basename(kw['config_file'][icrit])
            name = name.replace('.config', '')
            kw['output_prefix'] = orig_output_prefix + '_' + name
        print('================== start job', icrit, '======================')
        print('output_prefix:', kw['output_prefix'])
        print('criteria:', criteria)
        print('bbspec:', criteria.bbspec)

        if kw['precache_splices']:
            print('precaching splices')
            merge_bblock = kw['merge_bblock']
            del kw['merge_bblock']
            kw['bbs'] = simple_search_dag(
                criteria, merge_bblock=None, precache_only=True, **kw
            )
            if kw['only_bblocks']:
                assert len(kw['bbs']) is len(kw['only_bblocks'])
                for i, bb in enumerate(kw['bbs']):
                    kw['bbs'][i] = [bb[kw['only_bblocks'][i]]]
                print('modified bblock numbers (--only_bblocks)')
                print('   ', [len(b) for b in kw['bbs']])
            kw['merge_bblock'] = merge_bblock
            if kw['precache_splices_and_quit']:
                return

        global _shared_ssdag
        if 'bbs' in kw and (len(kw['bbs']) > 2
                            or kw['bbs'][0] is not kw['bbs'][1]):

            ############3

            #

            # _shared_ssdag = simple_search_dag(
            #    criteria, print_edge_summary=True, **kw
            # )

            merge_bblock = kw['merge_bblock']
            del kw['merge_bblock']
            _shared_ssdag = simple_search_dag(
                criteria, merge_bblock=0, print_edge_summary=True, **kw
            )
            kw['merge_bblock'] = merge_bblock
            print('memuse for global _shared_ssdag:')
            _shared_ssdag.report_memory_use()

            ####

            #

        if _shared_ssdag:
            if not 'bbs' in kw:
                kw['bbs'] = _shared_ssdag.bbs
            assert len(_shared_ssdag.bbs) == len(kw['bbs'])
            for a, b in zip(_shared_ssdag.bbs, kw['bbs']):
                for aa, bb in zip(a, b):
                    assert aa is bb

        log = worms_main_each_mergebb(criteria, **kw)
        if kw['pbar']:
            print('======================== logs ========================')
            for msg in log:
                print(msg)
    print('======================== done ========================')


def worms_main_each_mergebb(
        criteria, precache_splices, merge_bblock, parallel, verbosity, bbs,
        pbar, only_merge_bblocks, merge_segment, **kw
):
    exe = util.InProcessExecutor()
    if parallel:
        exe = cf.ProcessPoolExecutor(max_workers=parallel)
    bbs_states = [[b._state for b in bb] for bb in bbs]
    # kw['db'][0].clear_bblocks()  # remove cached BBlocks
    kw['db'][0].clear()
    kw['db'][1].clear()

    with exe as pool:
        mseg = merge_segment
        if mseg is None:
            mseg = criteria.merge_segment(**kw)
        if mseg is None:
            mseg = 0
        merge_bblock_list = range(len(bbs[mseg]))
        if only_merge_bblocks:
            merge_bblock_list = only_merge_bblocks
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
                **kw
            ) for i in merge_bblock_list
        ]
        log = [f'split job over merge_segment={mseg}, n = {len(futures)}']
        print(log[-1])

        fiter = cf.as_completed(futures)
        for f in fiter:
            log.extend(f.result())
        if pbar and log:
            log = [''] * len(futures) + log
        return log


def worms_main_protocol(
        criteria, bbs_states=None, disable_clash_check=0, **kw
):

    try:
        if bbs_states is not None:
            kw['bbs'] = [tuple(_BBlock(*s) for s in bb) for bb in bbs_states]

        ssdag, result1, log = search_func(criteria, **kw)
        if result1 is None: return []

        if disable_clash_check:
            result2 = result1
        else:
            result2 = prune_clashes(ssdag, criteria, result1, **kw)

        result3 = check_geometry(ssdag, criteria, result2, **kw)

        log = []
        if True:  # len(result3.idx) > 0:
            msg = f'nresults after clash/geom check {len(result3.idx):,}'
            log.append('    ' + msg)
            print(log[-1])

        log += filter_and_output_results(criteria, ssdag, result3, **kw)

        if not kw['pbar']:
            print(f'completed: mbb{kw["merge_bblock"]:04}')
            sys.stdout.flush()

        return log

    except Exception as e:
        print('error on mbb' + str(kw['merge_bblock']))
        print(type(e))
        print(traceback.format_exc())
        print(e)
        sys.stdout.flush()
        return []


def search_func(criteria, bbs, monte_carlo, merge_segment, **kw):

    stages = [criteria]
    if hasattr(criteria, 'stages'):
        stages = criteria.stages(bbs=bbs, **kw)
    if len(stages) > 1:
        assert kw['merge_bblock'] is not None

    assert len(monte_carlo) in (1, len(stages))
    if len(monte_carlo) != len(stages):
        monte_carlo *= len(stages)

    results = list()
    for i, stage in enumerate(stages):
        crit, stage_bbs = stage
        if callable(crit): crit = crit(*results[-1][:-1])
        lbl = f'stage{i}'
        if kw['merge_bblock'] is not None:
            lbl = f'stage{i}_mbb{kw["merge_bblock"]:04}'
        results.append(
            search_single_stage(
                crit,
                monte_carlo=monte_carlo[i],
                lbl=lbl,
                bbs=stage_bbs,
                merge_segment=merge_segment,
                **kw
            )
        )
        if (not hasattr(crit, 'produces_no_results')
                and len(results[-1][2].idx) is 0):
            print('mbb', kw['merge_bblock'], 'no results at stage', i)
            return None, None, None

    # todo: this whole block is very protocol-specific... needs refactoring
    if len(results) is 1:
        return results[0][1:]
    elif len(results) is 2:

        mseg = merge_segment
        if mseg is None:
            mseg = criteria.merge_segment(**kw)
        # simple_search_dag getting not-to-simple maybe split?
        _____, ssdA, rsltA, logA = results[0]
        critB, ssdB, rsltB, logB = results[1]
        assert _shared_ssdag
        ssdag = simple_search_dag(
            criteria,
            only_seg=mseg,
            make_edges=False,
            source=_shared_ssdag,
            bbs=bbs,
            **kw
        )
        ssdag.verts = ssdB.verts[:-1] + (ssdag.verts[mseg], ) + ssdA.verts[1:]

        assert len(ssdag.verts) == len(criteria.bbspec)
        rslt = merge_results_concat(
            criteria, ssdag, ssdA, rsltA, critB, ssdB, rsltB, **kw
        )
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
            **kw
        )
        ssdag.verts = ssdC.verts[:-1] + (ssdag.verts[mseg], ) + ssdB.verts[1:]
        assert len(ssdag.verts) == len(criteria.bbspec)

        rslt = merge_results_bridge(
            criteria, critC, ssdag, ssdB, ssdC, rsltC, **kw
        )

        return ssdag, rslt, logA + logB + logC

    else:

        assert 0, 'unknown 3+ stage protcol'


def search_single_stage(criteria, lbl='', **kw):

    if kw['run_cache']:
        if os.path.exists(kw['run_cache'] + lbl + '.pickle'):
            with (open(kw['run_cache'] + lbl + '.pickle', 'rb')) as inp:
                ssdag, result = _pickle.load(inp)
                return criteria, ssdag, result, ['from run cache ' + lbl]

    assert _shared_ssdag
    ssdag = simple_search_dag(criteria, source=_shared_ssdag, lbl=lbl, **kw)
    # print('number of bblocks:', [len(x) for x in ssdag.bbs])

    result, tsearch = run_and_time(
        grow_linear,
        ssdag=ssdag,
        loss_function=criteria.jit_lossfunc(),
        last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
        lbl=lbl,
        **kw
    )

    Nsparse = result.stats.total_samples[0]
    Nsparse_rate = int(Nsparse / tsearch)

    log = []
    if len(result.idx):
        frac_redundant = result.stats.n_redundant_results[0] / len(result.idx)
        log = [
            f'grow_linear {lbl} done, nresults {len(result.idx):,}, ' +
            f'samp/sec {Nsparse_rate:,}, redundant ratio {frac_redundant}'
        ]
    if log: print(log[-1])

    if kw['run_cache']:
        with (open(kw['run_cache'] + lbl + '.pickle', 'wb')) as out:
            _pickle.dump((ssdag, result), out)

    return criteria, ssdag, result, log


def getmem():
    mem = psutil.Process(os.getpid()).memory_info().rss / 2**20
    return f'{int(mem):5}'


def filter_and_output_results(
        criteria, ssdag, result, output_from_pose, merge_bblock, db,
        output_symmetric, output_centroid, output_prefix, max_output,
        max_score0, rms_err_cut, no_duplicate_bases, output_only_AAAA, **kw
):
    sf = ros.core.scoring.ScoreFunctionFactory.create_score_function('score0')
    sfsym = ros.core.scoring.symmetry.symmetrize_scorefunction(sf)

    mbb = ''
    if merge_bblock is not None: mbb = f'_mbb{merge_bblock:04d}'

    head = f'{output_prefix}{mbb}'
    if mbb and output_prefix[-1] != '/': head += '_'

    if not merge_bblock:
        # do this once per run, at merge_bblock == 0 (or None)
        with open(head + '__HEADER.info', 'w') as info_file:
            info_file.write(
                'close_err close_rms score0 score0sym filter zheight zradius radius nc nc_wo_jct n_nb  Name chain_info [exit_pdb exit_resN entrance_resN entrance_pdb]   jct_res \n'
            )

        # make json files with bblocks for single result
        # tmp, seenit = list(), set()
        # for j in range(len(ssdag.verts)):
        #     v = ssdag.verts[j]
        #     ibb = v.ibblock[result.idx[iresult, j]]
        #     bb = ssdag.bbs[j][ibb]
        #     fname = str(bytes(bb.file), 'utf-8')
        #     if fname not in seenit:
        #         for e in db[0]._alldb:
        #             if e['file'] == fname:
        #                 tmp.append(e)
        #     seenit.add(fname)
        # import json
        # with open('tmp_%i.json' % iresult, 'w') as out:
        #     json.dump(tmp, out)

    if output_from_pose:
        info_file = None
        nresults = 0
        Ntotal = min(max_output, len(result.idx))
        for iresult in range(Ntotal):

            # print(getmem(), 'MEM ================ top of loop ===============')

            if iresult % 100 == 0:
                process = psutil.Process(os.getpid())
                gc.collect()
                mem_before = process.memory_info().rss / float(2**20)
                pym_before = asizeof(db[0]) / float(2**20)
                db[0].clear()
                pym_after = asizeof(db[0]) / float(2**20)
                gc.collect()
                mem_after = process.memory_info().rss / float(2**20)
                print(
                    'clear db', mem_before, mem_after, mem_before - mem_after,
                    'pympler', pym_before, pym_after, pym_before - pym_after
                )

            if iresult % 10 == 0:
                process = psutil.Process(os.getpid())
                if hasattr(db[0], '_poses_cache'):
                    print(
                        f'mbb{merge_bblock:04} dumping results {iresult} of {Ntotal}',
                        'pose_cache', sys.getsizeof(db[0]._poses_cache),
                        len(db[0]._poses_cache),
                        f'{process.memory_info().rss / float(2**20):,}mb'
                    )

            bases = ssdag.get_bases(result.idx[iresult])
            bases_str = ','.join(bases)
            if no_duplicate_bases:
                if criteria.is_cyclic: bases = bases[:-1]
                if '' in bases: bases.remove('')
                if '?' in bases: bases.remove('?')
                if 'n/a' in bases: bases.remove('n/a')
                bases_uniq = set(bases)
                nbases = len(bases)
                if len(bases_uniq) != nbases:
                    if criteria.is_cyclic:
                        bases[-1] = '(' + bases[-1] + ')'
                    print('duplicate bases fail', merge_bblock, iresult, bases)
                    continue

            # print(getmem(), 'MEM make_pose_crit before')
            pose, prov = make_pose_crit(
                db[0],
                ssdag,
                criteria,
                result.idx[iresult],
                result.pos[iresult],
                only_connected='auto',
                provenance=True,
            )
            # print(getmem(), 'MEM make_pose_crit after')

            # print(getmem(), 'MEM dbfilters before')
            try:
                (jstr, jstr1, filt, grade, sp, mc, mcnh, mhc, nc, ncnh,
                 nhc) = run_db_filters(
                     db, criteria, ssdag, iresult, result.idx[iresult], pose,
                     prov, **kw
                 )
            except Exception as e:
                print('error in db_filters:')
                print(traceback.format_exc())
                print(e)
                continue
            # print(getmem(), 'MEM dbfilters after')

            if output_only_AAAA and grade != 'AAAA':
                # print(f'mbb{merge_bblock:04} {iresult:06} bad grade', grade)
                continue

            # print(getmem(), 'MEM rms before')
            rms = criteria.iface_rms(pose, prov, **kw)
            # if rms > rms_err_cut: continue
            # print(getmem(), 'MEM rms after')

            # print(getmem(), 'MEM poses and score0 before')
            cenpose = pose.clone()
            ros.core.util.switch_to_residue_type_set(cenpose, 'centroid')
            score0 = sf(cenpose)
            # print(getmem(), 'MEM poses and score0 after')
            if score0 > max_score0:
                print(
                    f'mbb{merge_bblock:04} {iresult:06} score0 fail',
                    merge_bblock, iresult, 'score0', score0, 'rms', rms,
                    'grade', grade
                )
                continue

            if hasattr(criteria, 'symfile_modifiers'):
                symdata = util.get_symdata_modified(
                    criteria.symname,
                    **criteria.symfile_modifiers(segpos=result.pos[iresult])
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
                # print(getmem(), 'MEM poses and score0sym after')

                if score0sym >= 2.0 * max_score0:
                    print(
                        f'mbb{merge_bblock:06} {iresult:04} score0sym fail',
                        score0sym, 'rms', rms, 'grade', grade
                    )
                    continue
            else:
                score0sym = -1

            mbbstr = 'None'
            if merge_bblock is not None:
                mbbstr = f'{merge_bblock:4d}'

            # print(getmem(), 'MEM chains before')
            chains = pose.split_by_chain()
            chain_info = '%4d ' % (len(list(chains)))
            chain_info += '-'.join(str(len(c)) for c in chains)
            # print(getmem(), 'MEM chains after')

            # print(getmem(), 'MEM get_affected_positions before')
            mod, new, lost, junct = get_affected_positions(cenpose, prov)
            # print(getmem(), 'MEM get_affected_positions after')

            jpos = '-'.join(str(x) for x in junct)
            fname = '%s_%04i_%s_%s_%s' % (
                head, iresult, jpos, jstr[:200], grade
            )

            # report bblock ids, taking into account merge_bblock shenani
            ibblock_list = [
                str(v.ibblock[i])
                for i, v in zip(result.idx[iresult], ssdag.verts)
            ]
            mseg = kw['merge_segment']
            mseg = criteria.merge_segment(**kw) if mseg is None else mseg
            mseg = mseg or 0  # 0 if None
            ibblock_list[mseg] = str(merge_bblock)

            if not info_file:
                info_file = open(f'{output_prefix}{mbb}.info', 'w')
            info_file.write(
                '%5.2f %5.2f %7.2f %7.2f %-8s %5.1f %5.1f %5.1f %4d %4d %4d %s %-80s %s  %s %s %s\n'
                % (
                    result.err[iresult], rms, score0, score0sym, grade,
                    result.zheight[iresult], result.zradius[iresult],
                    result.radius[iresult], mc, mcnh, mhc, bases_str, fname,
                    chain_info, jstr1, sp, '-'.join(ibblock_list)
                )
            )
            info_file.flush()

            # print(getmem(), 'MEM dump pdb before')
            if symdata and output_symmetric:
                sympose.dump_pdb(fname + '_sym.pdb')
            if output_centroid: pose = cenpose
            pose.dump_pdb(fname + '_asym.pdb')
            nresults += 1
            commas = lambda l: ','.join(str(_) for _ in l)
            with open(fname + '_asym.pdb', 'a') as out:
                for ip, p in enumerate(prov):
                    lb, ub, psrc, lbsrc, ubsrc = p
                    out.write(
                        f'Segment: {ip:2} resis {lb:4}-{ub:4} come from resis '
                        + f'{lbsrc}-{ubsrc} of {psrc.pdb_info().name()}\n'
                    )
                nchain = pose.num_chains()
                out.write('Bases: ' + bases_str + '\n')
                out.write('Modified positions: ' + commas(mod) + '\n')
                out.write('New contact positions: ' + commas(new) + '\n')
                out.write('Lost contact positions: ' + commas(lost) + '\n')
                out.write('Junction residues: ' + commas(junct) + '\n')
                out.write('Length of asymetric unit: ' + str(len(pose)) + '\n')
                out.write('Number of chains in ASU: ' + str(nchain) + '\n')
                out.write('Closure error: ' + str(rms) + '\n')
            # print(getmem(), 'MEM dump pdb after')

        if info_file is not None:
            info_file.close()

    else:
        nresults = 0
        for iresult in range(min(max_output, len(result.idx))):
            fname = '%s_%04i' % (head, iresult)
            print(result.err[iresult], fname)
            graph_dump_pdb(
                fname + '.pdb',
                ssdag,
                result.idx[iresult],
                result.pos[iresult],
                join='bb',
                trim=True
            )
            nresults += 1

    if nresults:
        return ['nresults output' + str(nresults)]
    else:
        return []


def merge_results_concat(
        criteria, ssdag, ssdagA, rsltA, critB, ssdagB, rsltB, merged_err_cut,
        max_merge, **kw
):
    bsfull = [x[0] for x in ssdag.bbspec]
    bspartA = [x[0] for x in ssdagA.bbspec]
    bspartB = [x[0] for x in ssdagB.bbspec]
    assert bsfull[-len(bspartA):] == bspartA
    assert bsfull[:len(bspartB)] == bspartB

    # print('merge_results_concat ssdag.bbspec', ssdag.bbspec)
    # print('merge_results_concat criteria.bbspec', criteria.bbspec)
    rsltB = subset_result(rsltB, slice(max_merge))

    binner = critB.binner
    hash_table = critB.hash_table
    from_seg = criteria.from_seg

    assert len(ssdagB.bbs[-1]) == len(ssdagA.bbs[0])
    assert len(ssdagB.bbs[-1]) == len(ssdag.bbs[from_seg])
    assert len(ssdagB.bbs[-1]) == 1, 'did you set merge_bblock?'
    assert ssdagB.bbs[-1][0].filehash == ssdagA.bbs[0][0].filehash
    assert ssdagB.bbs[-1][0].filehash == ssdag.bbs[from_seg][0].filehash
    for _ in range(from_seg):
        f = [bb.filehash for bb in ssdag.bbs[_]]
        assert f == [bb.filehash for bb in ssdagB.bbs[_]]
    for _ in range(len(ssdag.verts) - from_seg):
        f = [bb.filehash for bb in ssdag.bbs[from_seg + _]]
        assert f == [bb.filehash for bb in ssdagA.bbs[_]]

    n = len(rsltB.idx)
    nv = len(ssdag.verts)
    merged = ResultJIT(
        pos=np.empty((n, nv, 4, 4), dtype='f4'),
        idx=np.empty((n, nv), dtype='i4'),
        err=9e9 * np.ones((n, ), dtype='f8'),
        stats=np.empty(n, dtype='i4')
    )
    ok = np.ones(n, dtype=np.bool)
    for i_in_rslt in range(n):
        # print(rsltB.pos[i_in_rslt, -1])
        val = _get_hash_val(
            binner, hash_table, rsltB.pos[i_in_rslt, -1], criteria.nfold
        )
        # print(
        # 'merge_results_concat', i_in_rslt, val, np.right_shift(val, 32),
        # np.right_shift(val, 16) % 16,
        # np.right_shift(val, 8) % 8, val % 8
        # )
        if val < 0:
            print('val < 0')
            ok[i_in_rslt] = False
            continue
        i_ot_rslt = np.right_shift(val, 32)
        assert i_ot_rslt < len(rsltA.idx)

        # check score asap
        pos = np.concatenate((
            rsltB.pos[i_in_rslt, :-1],
            rsltB.pos[i_in_rslt, -1] @ rsltA.pos[i_ot_rslt, :]
        ))
        assert np.allclose(pos[from_seg], rsltB.pos[i_in_rslt, -1])
        err = criteria.score(pos.reshape(-1, 1, 4, 4))
        merged.err[i_in_rslt] = err
        # print('merge_results_concat', i_in_rslt, pos)
        # print('merge_results_concat', i_in_rslt, err)
        if err > merged_err_cut: continue

        i_outer = rsltA.idx[i_ot_rslt, 0]
        i_outer2 = rsltA.idx[i_ot_rslt, -1]
        i_inner = rsltB.idx[i_in_rslt, -1]
        v_inner = ssdagB.verts[-1]
        v_outer = ssdagA.verts[0]
        ibb = v_outer.ibblock[i_outer]
        assert ibb == 0
        ires_in = v_inner.ires[i_inner, 0]
        ires_out = v_outer.ires[i_outer, 1]
        isite_in = v_inner.isite[i_inner, 0]
        isite_out = v_outer.isite[i_outer, 1]
        isite_out2 = ssdagA.verts[-1].isite[i_outer2, 0]
        mrgv = ssdag.verts[from_seg]
        assert max(mrgv.ibblock) == 0
        assert max(ssdagA.verts[-1].ibblock) == 0

        imerge = util.binary_search_pair(mrgv.ires, (ires_in, ires_out))
        if imerge == -1:
            # if imerge < 0:
            ok[i_in_rslt] = False
            continue
        idx = np.concatenate(
            (rsltB.idx[i_in_rslt, :-1],
             [imerge], rsltA.idx[i_ot_rslt, 1:])
        )
        assert len(idx) == len(ssdag.verts)
        for ii, v in zip(idx, ssdag.verts):
            if v is not None: assert ii < v.len
        assert len(pos) == len(idx) == nv
        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.stats[i_in_rslt] = i_ot_rslt
    # print(merged.err[:100])
    nbad = np.sum(1 - ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    # print('bad score', np.sum(merged.err > merged_err_cut), 'of', n)
    ok[merged.err > merged_err_cut] = False
    ok = np.where(ok)[0][np.argsort(merged.err[ok])]
    merged = subset_result(merged, ok)
    return merged
