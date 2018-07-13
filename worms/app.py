import sys
import os
import argparse
import _pickle
from copy import deepcopy
from time import time

from xbin import gu_xbin_indexer, numba_xbin_indexer
import homog as hg

from worms.criteria import *
from worms.database import BBlockDB, SpliceDB
from worms.graph import linear_graph, graph_dump_pdb
from worms.search import grow_linear, SearchResult, subset_result
from worms.graph_pose import make_pose_crit
from worms.util import get_symdata
from worms.filters.clash import prune_clashes
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get

import pyrosetta
from pyrosetta import rosetta as ros


def parse_args(argv):
    args = get_cli_args(
        argv=argv,
        geometry=[''],
        bbconn=[''],
        nbblocks=64,
        monte_carlo=0.0,
        parallel=1,
        cachedir='',
        run_cache='',
        verbosity=2,
        #
        cache_sync=0.003,
        hash_cart_resl=1.0,
        hash_ori_resl=5.0,
        #
        max_splice_rms=0.7,
        #
        min_radius=0.0,
        i_merge_bblock=0,
        merged_err_cut=2.0,
        #
        max_clash_check=10000,
        ca_clash_dis=4.0,
        #
        max_output=1000,
        output_pose=True,
        output_symmetric=False,
        output_prefix='./worms',
        output_centroid=False,
        dbfiles=['']
    )
    crit = eval(''.join(args.geometry))
    bb = args.bbconn[1::2]
    nc = args.bbconn[0::2]
    assert len(nc) == len(bb)
    assert crit.from_seg < len(bb)
    assert crit.to_seg < len(bb)
    if isinstance(crit, Cyclic) and crit.origin_seg is not None:
        assert crit.origin_seg < len(bb)
    crit.bbspec = list(list(x) for x in zip(bb, nc))
    return crit, vars(args)


def run_and_time(func, *args, **kw):
    t = time()
    rslt = func(*args, **kw)
    t = time() - t
    if isinstance(rslt, tuple) and not isinstance(rslt, SearchResult):
        return rslt + (t, )
    return rslt, t


def worms_main(argv):

    # read inputs
    criteria, kw = parse_args(argv)
    print('worms_main, args:')
    for k, v in kw.items():
        print('   ', k, v)
    pyrosetta.init('-mute all -beta')
    db = BBlockDB(**kw), SpliceDB(**kw)

    # search
    search_func = select_search_function(criteria)
    graph, result, tsearch = run_and_time(search_func, db, criteria, **kw)
    print(f'raw results: {len(result.idx):,}, in {int(tsearch)}s')

    # filter
    result, tclash = run_and_time(prune_clashes, graph, criteria, result, **kw)
    print(f'nresults {len(result.idx):,}, dumping')

    # dump results
    output_results(db, criteria, graph, result, **kw)


def select_search_function(criteria):
    if isinstance(criteria, Cyclic) and criteria.origin_seg is not None:
        return search_two_stage
    else:
        return search_one_stage


def search_one_stage(db, criteria, singlebb=[], lbl='', **kw):
    if kw['run_cache']:
        if os.path.exists(kw['run_cache'] + lbl + '.pickle'):
            with (open(kw['run_cache'] + lbl + '.pickle', 'rb')) as inp:
                return _pickle.load(inp)
    graph = linear_graph(
        bbspec=criteria.bbspec,
        db=db,
        singlebb=singlebb,
        which_single=kw['i_merge_bblock'],
        **kw
    )
    print('calling grow_linear')
    result = grow_linear(
        graph=graph,
        loss_function=criteria.jit_lossfunc(),
        last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
        **kw
    )
    if kw['run_cache']:
        with (open(kw['run_cache'] + lbl + '.pickle', 'wb')) as out:
            _pickle.dump((graph, result), out)
    return graph, result


def output_results(
        db, criteria, graph, result, output_pose, output_symmetric,
        output_centroid, output_prefix, max_output, **kw
):
    for i in range(min(max_output, len(result.idx))):
        fname = output_prefix + '_%03i.pdb' % i
        if output_pose:
            pose = make_pose_crit(
                db[0],
                graph,
                criteria,
                result.idx[i],
                result.pos[i],
                only_connected='auto'
            )
            if output_centroid:
                ros.core.util.switch_to_residue_type_set(pose, 'centroid')
            if output_symmetric:
                symdata = get_symdata(criteria.symname)
                ros.core.pose.symmetry.make_symmetric_pose(pose, symdata)
            pose.dump_pdb(fname)
        else:
            if output_symmetric:
                raise NotImplementedError('no symmetry w/o poses')
            graph_dump_pdb(
                fname,
                graph,
                result.idx[i],
                result.pos[i],
                join='bb',
                trim=True
            )


def search_two_stage(db, criteria, **kw):

    critA = stage_one_criteria(criteria, **kw)
    graphA, rsltA = search_one_stage(
        db, critA, lbl='A', singlebb=[0, -1], **kw
    )

    critB = stage_two_criteria(criteria, graphA, rsltA, **kw)
    graphB, rsltB = search_one_stage(db, critB, lbl='B', singlebb=[-1], **kw)

    graph = linear_graph(
        bbspec=criteria.bbspec,
        db=db,
        singlebb=[criteria.from_seg, -1],
        which_single=kw['i_merge_bblock'],
        **kw
    )

    result, imerge = merge_results(
        criteria, graph, graphA, rsltA, critB, graphB, rsltB, **kw
    )

    return graph, result


def stage_one_criteria(criteria, min_radius=0, **kw):
    assert criteria.origin_seg == 0
    bbspec = deepcopy(criteria.bbspec[criteria.from_seg:])
    bbspec[0][1] = '_' + bbspec[0][1][1]
    if isinstance(criteria, Cyclic):
        critA = Cyclic(criteria.nfold, min_radius=min_radius)
    else:
        raise ValueError('dont know stage one for: ' + str(criteria))
    critA.bbspec = bbspec
    return critA


def stage_two_criteria(
        criteria, graphA, rsltA, hash_cart_resl, hash_ori_resl, **kw
):
    assert criteria.origin_seg == 0
    bbspec = deepcopy(criteria.bbspec[:criteria.from_seg + 1])
    bbspec[-1][1] = bbspec[-1][1][0] + '_'
    gubinner = gu_xbin_indexer(hash_cart_resl, hash_ori_resl)
    numba_binner = numba_xbin_indexer(hash_cart_resl, hash_ori_resl)
    keys, hash_table = _make_hash_table(graphA, rsltA, gubinner)
    critB = HashCriteria(criteria, numba_binner, hash_table)
    critB.bbspec = bbspec
    return critB


def get_cli_args(argv=None, **kw):
    if argv is None: argv = sys.argv[1:]
    # add from @files
    atfiles = []
    for a in argv:
        if a.startswith('@'):
            atfiles.append(a)
    for a in atfiles:
        argv.remove(a)
        with open(a[1:]) as inp:
            argv = list(inp.read().split()) + argv
    p = argparse.ArgumentParser()
    for k, v in kw.items():
        nargs = None
        type_ = type(v)
        if isinstance(v, list):
            nargs = '+'
            type_ = type(v[0])
        p.add_argument('--' + k, type=type_, dest=k, default=v, nargs=nargs)
        # print('arg', k, type_, nargs, v)
    args = p.parse_args(argv)
    return args


def _make_hash_table(graph, rslt, gubinner):
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    keys = gubinner(rslt.pos[:, -1])
    assert keys.dtype == np.int64
    ridx = np.arange(len(rslt.idx))
    ibb0 = graph.verts[+0].ibblock[rslt.idx[:, +0]]
    ibb1 = graph.verts[-1].ibblock[rslt.idx[:, -1]]
    isite0 = graph.verts[+0].isite[rslt.idx[:, +0], 1]
    isite1 = graph.verts[-1].isite[rslt.idx[:, -1], 0]
    assert np.all(ibb0 == ibb1)
    assert np.all(isite0 != isite1)
    assert np.all(isite0 < 2**8)
    assert np.all(isite1 < 2**8)
    assert np.all(ibb0 < 2**16)
    assert np.all(keys >= 0)
    vals = (
        np.left_shift(ridx, 32) + np.left_shift(ibb0, 16) +
        np.left_shift(isite0, 8) + isite1
    )
    hash_table = KHashi8i8()
    hash_table.update2(keys, vals)
    return keys, hash_table


def _get_has_lossfunc_data(nfold):
    rots = np.stack((
        hg.hrot([0, 0, 1, 0], np.pi * 2. / nfold),
        hg.hrot([0, 0, 1, 0], -np.pi * 2. / nfold),
    ))
    assert rots.shape == (2, 4, 4)
    irots = (0, 1) if nfold > 2 else (0, )
    return rots, irots


def _get_hash_val(gubinner, hash_table, pos, nfold):
    rots, irots = _get_has_lossfunc_data(nfold)
    for irot in irots:
        to_pos = rots[irot] @ pos
        xtgt = np.linalg.inv(pos) @ to_pos
        key = gubinner(xtgt)
        val = hash_table.get(key)
        if val != -9223372036854775808:
            return val
    assert 0, 'pos not found in table!'


class HashCriteria:
    def __init__(self, orig_criteria, binner, hash_table):
        self.orig_criteria = orig_criteria
        self.binner = binner
        self.hash_table = hash_table
        self.is_cyclic = False

    def jit_lossfunc(self):
        rots, irots = _get_has_lossfunc_data(self.orig_criteria.nfold)
        binner = self.binner
        hash_vp = self.table.hash

        @jit
        def func(pos, idx, verts):
            pos = pos[-1]
            for irot in irots:
                to_pos = rots[irot] @ pos
                xtgt = np.linalg.inv(pos) @ to_pos
                key = binner(xtgt)
                val = _khash_get(hash_vp, key, -9223372036854775808)

                # if missing, no hit
                if val == -9223372036854775808:
                    continue

                # must use same bblock
                ibody = verts[-1].ibblock[idx[-1]]
                ibody0 = np.right_shift(val, 16) % 2**16
                if ibody != ibody0:
                    continue

                # must use different site
                isite0 = np.right_shift(val, 8) % 2**8
                isite1 = val % 2**8
                isite = verts[-1].isite[idx[-1], 0]
                if isite == isite0 or isite == isite1:
                    continue

                return 0
            return 9e9

        return func


def merge_results(
        criteria, graph, graphA, rsltA, critB, graphB, rsltB, merged_err_cut,
        **kw
):
    binner = critB.binner
    hash_table = critB.hash_table
    assert len(graphB.bbs[-1]) == len(graphA.bbs[0]) == len(
        graph.bbs[criteria.from_seg]
    ) == 1
    assert graphB.bbs[-1][0].filehash == graphA.bbs[0][0].filehash
    assert graphB.bbs[-1][0].filehash == graph.bbs[criteria.from_seg
                                                   ][0].filehash
    for i in range(criteria.from_seg):
        [bb.filehash
         for bb in graph.bbs[i]] == [bb.filehash for bb in graphB.bbs[i]]
    for i in range(len(graph.verts) - criteria.from_seg):
        [bb.filehash for bb in graph.bbs[criteria.from_seg + i]
         ] == [bb.filehash for bb in graphA.bbs[i]]

    n = len(rsltB.idx)
    nv = len(graph.verts)
    merged = SearchResult(
        pos=np.empty((n, nv, 4, 4), dtype='f8'),
        idx=np.empty((n, nv), dtype='i4'),
        err=np.empty((n, ), dtype='f8'),
        stats=np.empty(n, dtype='i4')
    )
    ok = np.ones(n, dtype=np.bool)
    for i_in_rslt in range(n):
        val = _get_hash_val(
            binner, hash_table, rsltB.pos[i_in_rslt, -1], criteria.nfold
        )
        i_ot_rslt = np.right_shift(val, 32)
        assert i_ot_rslt < len(rsltA.idx)
        i_outer = rsltA.idx[i_ot_rslt, 0]
        i_outer2 = rsltA.idx[i_ot_rslt, -1]
        i_inner = rsltB.idx[i_in_rslt, -1]
        v_inner = graphB.verts[-1]
        v_outer = graphA.verts[0]
        ibb = v_outer.ibblock[i_outer]
        assert ibb == 0
        ires_in = v_inner.ires[i_inner, 0]
        ires_out = v_outer.ires[i_outer, 1]
        isite_in = v_inner.isite[i_inner, 0]
        isite_out = v_outer.isite[i_outer, 1]
        isite_out2 = graphA.verts[-1].isite[i_outer2, 0]
        mrgv = graph.verts[criteria.from_seg]
        assert max(mrgv.ibblock) == 0
        assert max(graphA.verts[-1].ibblock) == 0
        x = ((mrgv.ires[:, 0] == ires_in) * (mrgv.ires[:, 1] == ires_out))
        imerge = np.where(x)[0]
        # print(
        # ' ', i_in_rslt, ibb, ires_in, ires_out, isite_in, isite_out,
        # isite_out2, imerge
        # )
        if not len(imerge) == 1:
            ok[i_in_rslt] = False
            continue
        idx = np.concatenate(
            (rsltB.idx[i_in_rslt, :-1], imerge, rsltA.idx[i_ot_rslt, 1:])
        )
        assert len(idx) == len(graph.verts)
        for ii, v in zip(idx, graph.verts):
            assert ii < v.len
        pos = np.concatenate((
            rsltB.pos[i_in_rslt, :-1],
            rsltB.pos[i_in_rslt, -1] @ rsltA.pos[i_ot_rslt, :]
        ))
        assert np.allclose(pos[criteria.from_seg], rsltB.pos[i_in_rslt, -1])
        assert len(pos) == len(idx) == nv
        err = criteria.score(pos.reshape(-1, 1, 4, 4))

        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.err[i_in_rslt] = err
        merged.stats[i_in_rslt] = i_ot_rslt
    # print(merged.err[:100])
    nbad = np.sum(1 - ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    print('bad score', np.sum(merged.err > merged_err_cut), 'of', n)
    ok[merged.err > merged_err_cut] = False
    ok = np.where(ok)[0][np.argsort(merged.err[ok])]
    merged = subset_result(merged, ok)
    return merged, ok
