import logging
import sys
import glob
from time import clock, time
import pyrosetta
import glob
import _pickle as pickle

import os
from copy import deepcopy

import numpy as np
import pytest
from xbin import gu_xbin_indexer, numba_xbin_indexer
from homog import hrot, axis_angle_of

from worms import simple_search_dag, Cyclic, grow_linear, util
from worms.database import CachingBBlockDB, CachingSpliceDB
from worms.ssdag_pose import make_pose, make_pose_crit
from worms.ssdag import graph_dump_pdb
from worms.filters.clash import prune_clashes
from worms.util import jit
from worms.cli import get_cli_args
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.search import lossfunc_rand_1_in, subset_result, ResultJIT
from worms.bblock import bblock_dump_pdb

logging.getLogger().setLevel(99)

from pyrosetta import rosetta as ros


def main():

    args = get_cli_args(
        verbosity=2,
        parallel=1,
        nbblocks=4,
        dump_pdb=0,
        clash_check=0,
        cache_sync=0.003,
        monte_carlo=0.0,
        cart_resl=1.0,
        ori_resl=5.0,
        splice_max_rms=0.7,
        min_radius=0.0,
        i_merge_bblock=0,
        dbfiles=['must_specify_dbfile.json'],
    )

    pyrosetta.init('-mute all -beta')

    db = (CachingBBlockDB(**vars(args)), CachingSpliceDB(**vars(args)))

    spec = (
        [
            ['C3_N', '_N'],
            ['Het:NCy', 'CN'],
            ['Het:CCC', 'CC'],
            ['Het:NNy', 'NN'],
            ['Het:CCC', 'C_'],
        ],
        Cyclic(3, from_seg=2, origin_seg=0, min_radius=args.min_radius),
    )

    make_peace(spec=spec, db=db, **vars(args))


def make_peace(spec, cart_resl, ori_resl, clash_check, dump_pdb, **kw):
    binner = gu_xbin_indexer(cart_resl, ori_resl)
    numba_binner = numba_xbin_indexer(cart_resl, ori_resl)
    bbdb = kw['db'][0]
    bbspec, crit = spec

    ################ outer cycle ###############

    touter = time()
    ot_graph, ot_rslt, ot_crit = outside_grow(spec, **kw)
    rescore = ot_crit.score(ot_rslt.pos.swapaxes(0, 1))
    assert np.max(rescore[rescore < 9999]) <= 1.0
    ot_rslt = subset_result(ot_rslt, rescore <= 1.0)
    ntot = len(ot_rslt.idx)
    keys, hash_table = _make_hash_table(ot_graph, ot_rslt, binner)
    print(
        '  nresult outer', len(ot_rslt.idx), 'unique hashes', len(set(keys)),
        f'{int(ot_rslt.stats.total_samples[0] / (time() - touter)):,}/s',
        f'redundancy {ot_rslt.stats.n_redundant_results[0]/len(ot_rslt.idx):5.2f}'
    )

    ################ spokes ###############

    tinner = time()
    in_graph, in_rslt = inside_grow(
        spec, binner=numba_binner, table=hash_table, **kw
    )
    print(
        '  nresults inner', len(in_rslt.idx),
        f'rate {int(in_rslt.stats.total_samples[0] / (time() - tinner)):,}/s'
        f'redundancy {in_rslt.stats.n_redundant_results[0]/len(in_rslt.idx):5.2f}'
    )

    ################ merged ###############

    ssdag = simple_search_dag(
        bbspec,
        modbbs=modsinglebb((spec[1].from_seg, ), kw['i_merge_bblock']),
        make_edges=False,
        **kw
    )
    print('whole:', spec[0])
    rslt, imerge = merge_results(
        ssdag, crit, in_rslt, in_graph, ot_rslt, ot_graph, binner, hash_table
    )
    ntot = len(rslt.idx)
    tclash = time()
    rslt = prune_clashes(ssdag, crit, rslt, at_most=10000, thresh=3.0, **kw)
    tclash = time() - tclash
    print(
        '  nresults', len(rslt.idx), 'withclash', ntot, 'clashrate',
        ntot / tclash
    )

    symdata = util.get_symdata('C' + str(crit.nfold))
    for i in range(min(999, len(rslt.idx))):
        pose = make_pose_crit(
            bbdb, ssdag, crit, rslt.idx[i], rslt.pos[i], only_connected='auto'
        )
        ros.core.util.switch_to_residue_type_set(pose, 'centroid')
        ros.core.pose.symmetry.make_symmetric_pose(pose, symdata)
        pose.dump_pdb('whole_%03i.pdb' % i)

        # graph_dump_pdb('inner_%03i_bb.pdb' % i, in_graph, in_rslt.idx[imerge][i],
        # in_rslt.pos[imerge][i], join='bb', trim=False) # yapf: disable
        # graph_dump_pdb('inner_%03i_bb_trim.pdb' % i, in_graph, in_rslt.idx[imerge][i],
        # in_rslt.pos[imerge][i], join='bb', trim=True) # yapf: disable


        # i_ot_rslt = rslt.stats[i]
        # graph_dump_pdb('outer_%03i_bb.pdb' % i, ot_graph, ot_rslt.idx[i_ot_rslt],
        # ot_rslt.pos[i_ot_rslt], join='bb', trim=False) # yapf: disable
        # graph_dump_pdb('outer_%03i_bb_trim.pdb' % i, ot_graph, ot_rslt.idx[i_ot_rslt],
        # ot_rslt.pos[i_ot_rslt], join='bb', trim=True) # yapf: disable

        # graph_dump_pdb('whole_%03i_bb.pdb' % i, ssdag, rslt.idx[i], rslt.pos[i],
        # join='bb', trim=True) # yapf: disable

        # bb1 = ssdag.bbs[1][ssdag.verts[1].ibblock[rslt.idx[i, 1]]]
        # print(bytes(bb1.file))
        # bblock_dump_pdb('bblock1_%03i_bb.pdb' % i, bb1)






def merge_results(
        ssdag,
        crit,
        in_rslt,
        in_graph,
        ot_rslt,
        ot_graph,
        binner,
        hash_table,
        err_cut=2.0
):
    assert len(in_graph.bbs[-1]) == len(ot_graph.bbs[0]
                                        ) == len(ssdag.bbs[crit.from_seg]) == 1
    assert in_graph.bbs[-1][0].filehash == ot_graph.bbs[0][0].filehash
    assert in_graph.bbs[-1][0].filehash == ssdag.bbs[crit.from_seg][0].filehash
    for i in range(crit.from_seg):
        [bb.filehash
         for bb in ssdag.bbs[i]] == [bb.filehash for bb in in_graph.bbs[i]]
    for i in range(len(ssdag.verts) - crit.from_seg):
        [bb.filehash for bb in ssdag.bbs[crit.from_seg + i]
         ] == [bb.filehash for bb in ot_graph.bbs[i]]

    n = len(in_rslt.idx)
    nv = len(ssdag.verts)
    merged = ResultJIT(
        pos=np.empty((n, nv, 4, 4), dtype='f4'),
        idx=np.empty((n, nv), dtype='i4'),
        err=np.empty((n, ), dtype='f8'),
        stats=np.empty(n, dtype='i4')
    )
    ok = np.ones(n, dtype=np.bool)
    for i_in_rslt in range(n):
        val = _get_hash_val(
            binner, hash_table, in_rslt.pos[i_in_rslt, -1], crit.nfold
        )
        i_ot_rslt = np.right_shift(val, 32)
        assert i_ot_rslt < len(ot_rslt.idx)
        i_outer = ot_rslt.idx[i_ot_rslt, 0]
        i_outer2 = ot_rslt.idx[i_ot_rslt, -1]
        i_inner = in_rslt.idx[i_in_rslt, -1]
        v_inner = in_graph.verts[-1]
        v_outer = ot_graph.verts[0]
        ibb = v_outer.ibblock[i_outer]
        assert ibb == 0
        ires_in = v_inner.ires[i_inner, 0]
        ires_out = v_outer.ires[i_outer, 1]
        isite_in = v_inner.isite[i_inner, 0]
        isite_out = v_outer.isite[i_outer, 1]
        isite_out2 = ot_graph.verts[-1].isite[i_outer2, 0]
        mrgv = ssdag.verts[crit.from_seg]
        assert max(mrgv.ibblock) == 0
        assert max(ot_graph.verts[-1].ibblock) == 0
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
            (in_rslt.idx[i_in_rslt, :-1], imerge, ot_rslt.idx[i_ot_rslt, 1:])
        )
        assert len(idx) == len(ssdag.verts)
        for ii, v in zip(idx, ssdag.verts):
            assert ii < v.len
        pos = np.concatenate((
            in_rslt.pos[i_in_rslt, :-1],
            in_rslt.pos[i_in_rslt, -1] @ ot_rslt.pos[i_ot_rslt, :]
        ))
        assert np.allclose(pos[crit.from_seg], in_rslt.pos[i_in_rslt, -1])
        assert len(pos) == len(idx) == nv
        err = crit.score(pos.reshape(-1, 1, 4, 4))

        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.err[i_in_rslt] = err
        merged.stats[i_in_rslt] = i_ot_rslt
    # print(merged.err[:100])
    nbad = np.sum(1 - ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    print('bad score', np.sum(merged.err > 2.0), 'of', n)
    ok[merged.err > 2.0] = False
    ok = np.where(ok)[0][np.argsort(merged.err[ok])]
    merged = subset_result(merged, ok)
    return merged, ok


def modsinglebb(iverts, ibb):
    def func(bbs):
        for i in iverts:
            bbs[i] = (bbs[i][ibb], )

    return func


def outside_grow(spec, min_radius, i_merge_bblock, **kw):
    bbspec0, crit0 = spec
    bbspec = deepcopy(bbspec0[crit0.from_seg:])
    bbspec[0][1] = '_' + bbspec[0][1][1]
    print('outside', bbspec)

    if os.path.exists('outer.pickle'):
        with open('outer.pickle', 'rb') as inp:
            ssdag, rslt, crit = pickle.load(inp)
    else:
        crit = Cyclic(crit0.nfold, min_radius=min_radius)
        ssdag = simple_search_dag(bbspec, modbbs=modsinglebb((0, -1), i_merge_bblock),
            **kw)
        rslt = grow_linear(
            ssdag,
            loss_function=crit.jit_lossfunc(),
            loss_threshold=1.0,
            last_bb_same_as=crit.from_seg,
            **kw
        )
        with open('outer.pickle', 'wb') as out:
            pickle.dump((ssdag, rslt, crit), out)

    return ssdag, rslt, crit


def inside_grow(spec, binner, table, i_merge_bblock, **kw):
    bbspec0, crit0 = spec
    bbspec = deepcopy(bbspec0[:crit0.from_seg + 1])
    bbspec[-1][1] = bbspec[-1][1][0] + '_'
    print('inside', bbspec)

    if os.path.exists('inner.pickle'):
        with open('inner.pickle', 'rb') as inp:
            ssdag, rslt = pickle.load(inp)
    else:
        ssdag = simple_search_dag(bbspec, modbbs=modsinglebb((-1, ), i_merge_bblock),
            **kw)
        rslt = grow_linear(
            ssdag,
            loss_function=_hash_lossfunc(binner, table, crit0.nfold),
            # loss_function=lossfunc_rand_1_in(1000),
            loss_threshold=1.0,
            **kw
        )
        with open('inner.pickle', 'wb') as out:
            pickle.dump((ssdag, rslt), out)
    return ssdag, rslt



def _make_hash_table(ssdag, rslt, binner):
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    assert np.max(np.abs(rslt.pos[:, -1, 0, 3])) < 512.0
    keys = binner(rslt.pos[:, -1])
    assert keys.dtype == np.int64
    ridx = np.arange(len(rslt.idx))
    ibb0 = ssdag.verts[+0].ibblock[rslt.idx[:, +0]]
    ibb1 = ssdag.verts[-1].ibblock[rslt.idx[:, -1]]
    isite0 = ssdag.verts[+0].isite[rslt.idx[:, +0], 1]
    isite1 = ssdag.verts[-1].isite[rslt.idx[:, -1], 0]
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
        hrot([0, 0, 1, 0], np.pi * 2. / nfold),
        hrot([0, 0, 1, 0], -np.pi * 2. / nfold),
    ))
    assert rots.shape == (2, 4, 4)
    irots = (0, 1) if nfold > 2 else (0, )
    return rots, irots


def _get_hash_val(binner, hash_table, pos, nfold):
    rots, irots = _get_has_lossfunc_data(nfold)
    for irot in irots:
        to_pos = rots[irot] @ pos
        xtgt = np.linalg.inv(pos) @ to_pos
        key = binner(xtgt)
        val = hash_table.get(key)
        if val != -9223372036854775808:
            return val
    assert 0, 'pos not found in table!'


def _hash_lossfunc(binner, table, nfold):
    rots, irots = _get_has_lossfunc_data(nfold)
    hash_vp = table.hash

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


if __name__ == '__main__':
    main()
