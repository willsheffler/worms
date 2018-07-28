import sys
import os
import argparse
import _pickle
from copy import deepcopy

from xbin import gu_xbin_indexer, numba_xbin_indexer
import homog as hg

from worms.criteria import *
from worms.database import BBlockDB, SpliceDB
from worms.ssdag import simple_search_dag, graph_dump_pdb
from worms.search import grow_linear, SearchResult, subset_result
from worms.ssdag_pose import make_pose_crit
from worms.util import get_symdata, run_and_time, binary_search_pair, cpu_count, get_cli_args
from worms.filters.clash import prune_clashes
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.criteria.hash_util import _get_hash_val
from worms.filters.db_filters import get_affected_positions

import pyrosetta
from pyrosetta import rosetta as ros


def parse_args(argv):
    args = get_cli_args(
        argv=argv,
        geometry=[''],
        bbconn=[''],
        config_file='',
        nbblocks=64,
        monte_carlo=0.0,
        parallel=1,
        verbosity=2,
        precache_splices=False,
        #
        cachedirs=[''],
        dbfiles=[''],
        load_poses=False,
        read_new_pdbs=False,
        run_cache='',

        #
        cache_sync=0.003,
        hash_cart_resl=1.0,
        hash_ori_resl=5.0,
        #
        # splice stuff
        splice_max_rms=0.7,
        splice_ncontact_cut=30,
        splice_clash_d2=4.0**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_rms_range=6,
        splice_clash_contact_range=60,
        #
        min_radius=0.0,
        merge_bblock=-1,
        merged_err_cut=2.0,
        max_merge=10000,
        #
        max_clash_check=10000,
        ca_clash_dis=4.0,
        #
        max_output=1000,
        output_pose=True,
        output_symmetric=False,
        output_prefix='./worms',
        output_centroid=False,

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
    # kw['bblockdb'] = BBlockDB(**kw)
    # kw['splicedb'] = SpliceDB(**kw)

    return crit, kw


def worms_main(argv):

    # read inputs
    criteria, kw = parse_args(argv)
    print('worms_main, args:')
    for k, v in kw.items():
        print('   ', k, v)
    pyrosetta.init('-mute all -beta')

    # search
    ssdag = simple_search_dag(criteria, **kw)
    result, tsearch = run_and_time(search_func, criteria, ssdag, **kw)
    print(f'raw results: {len(result.idx):,}, in {int(tsearch)}s')

    # filter
    result, tclash = run_and_time(prune_clashes, ssdag, criteria, result, **kw)
    print(f'nresults {len(result.idx):,}, dumping')

    # dump results
    output_results(criteria, ssdag, result, **kw)


def search_func(criteria, ssdag, **kw):

    stages = [criteria]
    if hasattr(criteria, 'stages'):
        stages = criteria.stages(**kw)
    results = list()
    for i, stage_criteria in enumerate(stages):
        if callable(stage_criteria):
            stage_criteria = stage_criteria(*results[-1])
        results.append(search_single_stage(stage_criteria, lbl=str(i), **kw))

    if len(results) == 1:
        return results[0][-1]
    elif len(results) == 2:
        critA, ssdagA, rsltA = results[0]
        critB, ssdagB, rsltB = results[1]
        return merge_results(
            criteria, ssdag, ssdagA, rsltA, critB, ssdagB, rsltB, **kw
        )
    else:
        raise NotImplementedError('dunno more than two stages!')


def search_single_stage(criteria, lbl='', **kw):
    if kw['run_cache']:
        if os.path.exists(kw['run_cache'] + lbl + '.pickle'):
            with (open(kw['run_cache'] + lbl + '.pickle', 'rb')) as inp:
                ssdag, result = _pickle.load(inp)
                return criteria, ssdag, result
    ssdag = simple_search_dag(criteria, **kw)
    result = grow_linear(
        ssdag=ssdag,
        loss_function=criteria.jit_lossfunc(),
        last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
        **kw
    )
    print('grow_linear done, nresluts', len(result.idx))
    if kw['run_cache']:
        with (open(kw['run_cache'] + lbl + '.pickle', 'wb')) as out:
            _pickle.dump((ssdag, result), out)

    # print(result.idx[:10])
    # print(result.err[:100])
    # assert 0

    return criteria, ssdag, result


def output_results(
        criteria, ssdag, result, output_pose, merge_bblock, db,
        output_symmetric, output_centroid, output_prefix, max_output, **kw
):
    for i in range(min(max_output, len(result.idx))):

        # tmp, seenit = list(), set()
        # for j in range(len(ssdag.verts)):
        #     v = ssdag.verts[j]
        #     ibb = v.ibblock[result.idx[i, j]]
        #     bb = ssdag.bbs[j][ibb]
        #     fname = str(bytes(bb.file), 'utf-8')
        #     if fname not in seenit:
        #         for e in db[0]._alldb:
        #             if e['file'] == fname:
        #                 tmp.append(e)
        #     seenit.add(fname)
        # import json
        # with open('tmp_%i.json' % i, 'w') as out:
        #     json.dump(tmp, out)

        if merge_bblock is not None: mbb = f'_mbb{merge_bblock:03d}'
        fname = output_prefix + mbb + '_%03i.pdb' % i
        if output_pose:
            pose, prov = make_pose_crit(
                db[0],
                ssdag,
                criteria,
                result.idx[i],
                result.pos[i],
                only_connected='auto',
                provenance=True
            )
            with open('wip_pose_prov.pickle', 'wb') as out:
                _pickle.dump((pose, prov), out)
            if output_centroid:
                ros.core.util.switch_to_residue_type_set(pose, 'centroid')
            if output_symmetric:
                symdata = get_symdata(criteria.symname)
                ros.core.pose.symmetry.make_symmetric_pose(pose, symdata)

            print('output pdb', fname)
            mod, new, lost, junct = get_affected_positions(pose, prov)
            pose.dump_pdb(fname)
            commas = lambda l: ','.join(str(_) for _ in l)
            with open(fname, 'a') as out:
                nchain = pose.num_chains()
                out.write('Modified positions: ' + commas(mod) + '\n')
                out.write('New contact positions: ' + commas(new) + '\n')
                out.write('Lost contact positions: ' + commas(lost) + '\n')
                out.write('Junction residues: ' + commas(junct) + '\n')
                out.write('Length of asymetric unit: ' + str(len(pose)) + '\n')
                out.write('Number of chains in ASU: ' + str(nchain) + '\n')
                out.write('Closure error: ' + str(result.err[i]) + '\n')

        else:
            if output_symmetric:
                raise NotImplementedError('no symmetry w/o poses')
            graph_dump_pdb(
                fname,
                ssdag,
                result.idx[i],
                result.pos[i],
                join='bb',
                trim=True
            )


def merge_results(
        criteria, ssdag, ssdagA, rsltA, critB, ssdagB, rsltB, merged_err_cut,
        max_merge, **kw
):
    bsfull = [x[0] for x in ssdag.bbspec]
    bspartA = [x[0] for x in ssdagA.bbspec]
    bspartB = [x[0] for x in ssdagB.bbspec]
    assert bsfull[-len(bspartA):] == bspartA
    assert bsfull[:len(bspartB)] == bspartB

    # print('merge_results ssdag.bbspec', ssdag.bbspec)
    # print('merge_results criteria.bbspec', criteria.bbspec)
    rsltB = subset_result(rsltB, slice(max_merge))

    binner = critB.binner
    hash_table = critB.hash_table
    assert len(ssdagB.bbs[-1]) == len(ssdagA.bbs[0]) == len(
        ssdag.bbs[criteria.from_seg]
    ) == 1
    assert ssdagB.bbs[-1][0].filehash == ssdagA.bbs[0][0].filehash
    assert ssdagB.bbs[-1][0].filehash == ssdag.bbs[criteria.from_seg
                                                   ][0].filehash
    for _ in range(criteria.from_seg):
        f = [bb.filehash for bb in ssdag.bbs[_]]
        assert f == [bb.filehash for bb in ssdagB.bbs[_]]
    for _ in range(len(ssdag.verts) - criteria.from_seg):
        f = [bb.filehash for bb in ssdag.bbs[criteria.from_seg + _]]
        assert f == [bb.filehash for bb in ssdagA.bbs[_]]

    n = len(rsltB.idx)
    nv = len(ssdag.verts)
    merged = SearchResult(
        pos=np.empty((n, nv, 4, 4), dtype='f8'),
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
        # 'merge_results', i_in_rslt, val, np.right_shift(val, 32),
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
        assert np.allclose(pos[criteria.from_seg], rsltB.pos[i_in_rslt, -1])
        err = criteria.score(pos.reshape(-1, 1, 4, 4))
        merged.err[i_in_rslt] = err
        # print('merge_results', i_in_rslt, pos)
        # print('merge_results', i_in_rslt, err)
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
        mrgv = ssdag.verts[criteria.from_seg]
        assert max(mrgv.ibblock) == 0
        assert max(ssdagA.verts[-1].ibblock) == 0

        imerge = binary_search_pair(mrgv.ires, (ires_in, ires_out))
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
            assert ii < v.len
        assert len(pos) == len(idx) == nv
        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.stats[i_in_rslt] = i_ot_rslt
    # print(merged.err[:100])
    nbad = np.sum(1 - ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    print('bad score', np.sum(merged.err > merged_err_cut), 'of', n)
    ok[merged.err > merged_err_cut] = False
    ok = np.where(ok)[0][np.argsort(merged.err[ok])]
    merged = subset_result(merged, ok)
    return merged
