import sys
import os
import argparse
import _pickle
from copy import deepcopy
import itertools as it
from time import time
import concurrent.futures as cf
import traceback

from tqdm import tqdm
from xbin import gu_xbin_indexer, numba_xbin_indexer
import homog as hg

from worms.criteria import *
from worms.database import BBlockDB, SpliceDB
from worms.ssdag import simple_search_dag, graph_dump_pdb
from worms.search import grow_linear, SearchResult, subset_result
from worms.ssdag_pose import make_pose_crit
from worms.util import run_and_time
from worms import util
from worms.filters.clash import prune_clashes
from worms.filters.db_filters import run_db_filters
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.criteria.hash_util import _get_hash_val
from worms.filters.db_filters import get_affected_positions
from worms.bblock import _BBlock

import pyrosetta
from pyrosetta import rosetta as ros


def parse_args(argv):
    args = util.get_cli_args(
        argv=argv,
        geometry=[''],
        bbconn=[''],
        config_file='',
        nbblocks=64,
        monte_carlo=[0.0],
        parallel=1,
        verbosity=2,
        precache_splices=1,
        precache_splices_and_quit=0,
        pbar=0,
        pbar_interval=10.0,
        #
        cachedirs=[''],
        dbfiles=[''],
        load_poses=0,
        read_new_pdbs=0,
        run_cache='',
        merge_bblock=-1,
        no_duplicate_bases=1,
        shuffle_bblocks=1,

        # splice stuff
        splice_rms_range=5,
        splice_max_rms=0.6,
        splice_clash_d2=3.5**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_clash_contact_range=30,
        splice_ncontact_cut=30,
        #
        tol=1.0,
        lever=25.0,
        min_radius=0.0,
        hash_cart_resl=1.0,
        hash_ori_resl=5.0,
        merged_err_cut=999.0,
        rms_err_cut=3.0,
        ca_clash_dis=3.0,
        #
        max_merge=10000,
        max_clash_check=10000,
        max_output=1000,
        max_score0=10,
        #
        output_from_pose=1,
        output_symmetric=1,
        output_prefix='./worms',
        output_centroid=0,
        #
        cache_sync=0.003,
        #
        postfilt_splice_max_rms=0.6,
        postfilt_splice_rms_length=10,
        postfilt_splice_ncontact_cut=40,
        postfilt_splice_ncontact_no_helix_cut=2,
        postfilt_splice_nhelix_contacted_cut=3,

    )
    if not args.config_file:
        crit = eval(''.join(args.geometry))
        bb = args.bbconn[1::2]
        nc = args.bbconn[0::2]
    else:
        with open(args.config_file) as inp:
            lines = inp.readlines()
            assert len(lines) is 2

            def orient(a, b):
                return (a or '_') + (b or '_')

            bbnc = eval(lines[0])
            bb = [x[0] for x in bbnc]
            nc = [x[1] for x in bbnc]
            crit = eval(lines[1])

    assert len(nc) == len(bb)
    assert crit.from_seg < len(bb)
    assert crit.to_seg < len(bb)
    if isinstance(crit, Cyclic) and crit.origin_seg is not None:
        assert crit.origin_seg < len(bb)
    crit.bbspec = list(list(x) for x in zip(bb, nc))
    if args.merge_bblock < 0: args.merge_bblock = None
    kw = vars(args)
    kw['db'] = BBlockDB(**kw), SpliceDB(**kw)

    return crit, kw


_shared_ssdag = None


def worms_main(argv):

    # read inputs
    criteria, kw = parse_args(argv)
    print('worms_main, args:')
    for k, v in kw.items():
        print('   ', k, v)
    pyrosetta.init('-mute all -beta')

    #

    # with open('wip_db_filters.pickle', 'rb') as inp:
    #     ssdag, result, pose, prov = _pickle.load(inp)

    # db = kw['db']
    # del kw['db']
    # (jstr, jstr1, filter, grade, splices, mc, mcnh, mhc, nc, ncnh,
    #  nhc) = run_db_filters(
    #      db, criteria, ssdag, 0, result.idx[0], pose, prov, **kw
    #  )
    # print(jstr)
    # print(jstr1)
    # print(filter)
    # print(grade)
    # print(splices)
    # print(mc, nc)
    # print(mcnh, ncnh)
    # print(mhc, nhc)
    # sys.exit()

    if kw['precache_splices']:
        merge_bblock = kw['merge_bblock']
        del kw['merge_bblock']
        kw['bbs'] = simple_search_dag(
            criteria, merge_bblock=None, precache_only=True, **kw
        )
        kw['merge_bblock'] = merge_bblock
        if kw['precache_splices_and_quit']:
            return

    global _shared_ssdag
    _shared_ssdag = simple_search_dag(criteria, print_edge_summary=True, **kw)
    if not 'bbs' in kw:
        kw['bbs'] = _shared_ssdag.bbs
    assert len(_shared_ssdag.bbs) == len(kw['bbs'])
    for a, b in zip(_shared_ssdag.bbs, kw['bbs']):
        assert a is b
    if not kw['shuffle_bblocks']:
        bbnames = [[bytes(b.file).decode('utf-8')
                    for b in bb]
                   for bb in kw['bbs']]
        with open(kw['output_prefix'] + '_bblocks.pickle', 'wb') as out:
            _pickle.dump(bbnames, out)

    # merge_segment = criteria.merge_segment(**kw)
    if kw['parallel'] == 0:
        log = worms_main_protocol(criteria, **kw)
    else:
        log = worms_main_each_mergebb(criteria, **kw)
    if kw['pbar']:
        print('======================== logs ========================')
        for msg in log:
            print(msg)
        print('======================== done ========================')


def worms_main_each_mergebb(
        criteria, precache_splices, merge_bblock, parallel, verbosity, bbs,
        pbar, **kw
):
    exe = util.InProcessExecutor()
    if parallel:
        exe = cf.ProcessPoolExecutor(max_workers=parallel)
    bbs_states = [[b._state for b in bb] for bb in bbs]
    kw['db'][0].clear()  # remove cached BBlocks
    with exe as pool:
        merge_segment = criteria.merge_segment(**kw)
        if merge_segment is None:
            merge_segment = 0
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
                **kw
            ) for i in range(len(bbs[merge_segment]))
        ]
        log = ['parallel over merge_bblock, n =' + str(len(futures))]
        if not pbar: print(log[-1])

        fiter = cf.as_completed(futures)
        for f in fiter:
            log.extend(f.result())
        if pbar:
            log = [''] * len(futures) + log
        return log


def worms_main_protocol(criteria, bbs_states=None, **kw):

    if bbs_states is not None:
        kw['bbs'] = [tuple(_BBlock(*s) for s in bb) for bb in bbs_states]

    # search
    tup, tsearch = run_and_time(search_func, criteria, **kw)
    ssdag, result, log = tup
    # print(f'raw results: {len(result.idx):,}, in {int(tsearch)}s')

    filter
    result2, tclash = run_and_time(
        prune_clashes, ssdag, criteria, result, **kw
    )

    msg = f'nresults {len(result2.idx):,}, tot w/clashes {len(result.idx):,}'
    log.append('    ' + msg)
    print(log[-1])

    # dump results
    filter_and_output_results(criteria, ssdag, result2, **kw)

    return log


def search_func(criteria, bbs, monte_carlo, **kw):

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
                crit, monte_carlo=monte_carlo[i], lbl=lbl, bbs=stage_bbs, **kw
            )
        )

    if len(results) == 1:
        return results[0][1:]
    elif len(results) == 2:
        # todo: this whole block is very protocol-specific... needs refactoring
        mseg = criteria.merge_segment(**kw)
        # simple_search_dag getting not-to-simple maybe split?
        _____, ssdA, rsltA, logA = results[0]
        critB, ssdB, rsltB, logB = results[1]
        # remove unnecessary verts before creating more
        # ssdA.verts = ssdA.verts[:1] + (None, ) * len(ssdA.verts[1:])
        # ssdB.verts = (None, ) * len(ssdB.verts[:1]) + ssdB.verts[-1:]
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
    else:
        raise NotImplementedError('dunno more than two stages!')


def search_single_stage(criteria, lbl='', **kw):

    if kw['run_cache']:
        if os.path.exists(kw['run_cache'] + lbl + '.pickle'):
            with (open(kw['run_cache'] + lbl + '.pickle', 'rb')) as inp:
                ssdag, result = _pickle.load(inp)
                return criteria, ssdag, result, ['from run cache ' + lbl]

    ssdag = simple_search_dag(criteria, source=_shared_ssdag, lbl=lbl, **kw)

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
    frac_redundant = 0
    if len(result.idx):
        frac_redundant = result.stats.n_redundant_results[0] / len(result.idx)

    log = [
        f'grow_linear {lbl} done, nresults {len(result.idx):,}, ' +
        f'samp/sec {Nsparse_rate:,}, redundant ratio {frac_redundant}'
    ]
    if not kw['pbar']: print(log[-1])

    if kw['run_cache']:
        with (open(kw['run_cache'] + lbl + '.pickle', 'wb')) as out:
            _pickle.dump((ssdag, result), out)

    return criteria, ssdag, result, log


def filter_and_output_results(
        criteria, ssdag, result, output_from_pose, merge_bblock, db,
        output_symmetric, output_centroid, output_prefix, max_output,
        max_score0, rms_err_cut, **kw
):
    sf = ros.core.scoring.ScoreFunctionFactory.create_score_function('score0')
    sfsym = ros.core.scoring.symmetry.symmetrize_scorefunction(sf)
    mbb = ''
    if merge_bblock is not None: mbb = f'_mbb{merge_bblock:04d}'
    with open(f'{output_prefix}{mbb}.info', 'w') as info_file:
        for iresult in range(min(max_output, len(result.idx))):

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
                pose, prov = make_pose_crit(
                    db[0],
                    ssdag,
                    criteria,
                    result.idx[iresult],
                    result.pos[iresult],
                    only_connected='auto',
                    provenance=True
                )

                if iresult == 0:
                    chain_header = 'Nchains ' + ' '
                    for chain in range(pose.num_chains()):
                        chain_header = chain_header + ' L%s ' % (chain + 1)
                    info_file.write(
                        'close_err close_rms score0 filter nc nc_wo_jct n_nb  Name                                                                  %s   [exit_pdb       exit_resN entrance_resN entrance_pdb        ]   jct_res \n'
                        % chain_header
                    )

                # with open('wip_db_filters.pickle', 'wb') as out:
                # _pickle.dump((ssdag, result, pose, prov), out)

                try:
                    (
                        jstr, jstr1, filt, grade, sp, mc, mcnh, mhc, nc, ncnh,
                        nhc
                    ) = run_db_filters(
                        db, criteria, ssdag, iresult, result.idx[iresult],
                        pose, prov, **kw
                    )
                except Exception as e:
                    print('error in db_filters:')
                    print(traceback.format_exc())
                    print(exc)
                    continue

                head = f'{output_prefix}{mbb}_'
                fname = '%s_%04i_%s_%s_%s_%s_%s' % (
                    head, iresult, jstr.replace('_0001', '')[:200], grade, mc,
                    mcnh, mhc
                )

                rms = criteria.iface_rms(pose, prov, **kw)
                # if rms > rms_err_cut: continue

                cenpose = pose.clone()
                ros.core.util.switch_to_residue_type_set(cenpose, 'centroid')

                symdata = util.get_symdata(criteria.symname)
                sympose = cenpose.clone()
                ros.core.pose.symmetry.make_symmetric_pose(sympose, symdata)
                score0 = sfsym(sympose)
                # if score0 >= 10 * max_score0: continue

                bases = ssdag.get_bases(result.idx[iresult])
                # print(bases, ssdag.get_base_hashes(result.idx[iresult]))
                bases_str = ','.join(bases)
                mbbstr = 'None'
                if merge_bblock is not None:
                    mbbstr = f'{merge_bblock:4d}'
                print(
                    f'mbb{mbbstr} {iresult:4d} err {result.err[iresult]:5.2f} rms {rms:5.2f} score0 {score0:7.2f} {grade} {filt} {bases_str} {fname}'
                )
                # out_file.write('%-80s %s  %3.2f  %7.2f  %s %s %-8s %5.2f %4d %4d %4d \n'%(junct_str,chain_info,s[top_hit],score0,junct_str1,w.splicepoints(top_hit),filter,result,min_contacts,min_contacts_no_helix,min_helices_contacted))
                chains = pose.split_by_chain()
                chain_info = '%4d ' % (len(list(chains)))
                for chain in chains:
                    chain_info = chain_info + '%4d ' % chain.size()
                info_file.write(
                    '%5.2f %5.2f %7.2f %-8s %4d %4d %4d %s %-80s %s  %s %s \n'
                    % (
                        result.err[iresult], rms, score0, grade, mc, mcnh, mhc,
                        bases_str, fname, chain_info, jstr1, sp
                    )
                )
                info_file.flush()

                if score0 >= max_score0: continue
                mod, new, lost, junct = get_affected_positions(sympose, prov)
                if output_symmetric: sympose.dump_pdb(fname + '_sym.pdb')
                if output_centroid: pose = cenpose
                pose.dump_pdb(fname + '_asym.pdb')
                commas = lambda l: ','.join(str(_) for _ in l)
                with open(fname + '_asym.pdb', 'a') as out:
                    for ip, p in enumerate(prov):
                        lb, ub, psrc, lbsrc, ubsrc = p
                        out.write(
                            f'Segment: {ip:2} resis {lb:4}-{ub:4} come from resis'
                            + f'{lbsrc}-{ubsrc} of {psrc.pdb_info().name()}\n'
                        )
                    nchain = pose.num_chains()
                    out.write('Modified positions: ' + commas(mod) + '\n')
                    out.write('New contact positions: ' + commas(new) + '\n')
                    out.write('Lost contact positions: ' + commas(lost) + '\n')
                    out.write('Junction residues: ' + commas(junct) + '\n')
                    out.write(
                        'Length of asymetric unit: ' + str(len(pose)) + '\n'
                    )
                    out.write('Number of chains in ASU: ' + str(nchain) + '\n')
                    out.write('Closure error: ' + str(rms) + '\n')

            else:
                # if output_symmetric:
                # raise NotImplementedError('no symmetry w/o poses')
                head = f'{output_prefix}{mbb}_'
                fname = '%s_%04i' % (head, iresult)
                graph_dump_pdb(
                    fname + '.pdb',
                    ssdag,
                    result.idx[iresult],
                    result.pos[iresult],
                    join='bb',
                    trim=True
                )


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
    merged = SearchResult(
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
