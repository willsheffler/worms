import logging
import sys
import glob
from time import clock, time
import argparse
import pyrosetta
import glob
import _pickle as pickle
import os
from copy import deepcopy

import numpy as np
import pytest
from xbin import gu_xbin_indexer, numba_xbin_indexer
from homog import hrot

from worms import linear_gragh, Cyclic, grow_linear
from worms.database import BBlockDB, SpliceDB
from worms.graph_pose import make_pose, make_pose_crit
from worms.graph import graph_dump_pdb
from worms.filters.clash import prune_clashing_results
from worms.util import jit
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get
from worms.search import lossfunc_rand_1_in, SearchResult

logging.getLogger().setLevel(99)


def main():

    args = _get_args(
        verbosity=2,
        parallel=1,
        nbblocks=4,
        dump_pdb=0,
        clash_check=0,
        cache_sync=0.003,
        monte_carlo=0.0,
        cart_resl=1.0,
        ori_resl=5.0,
        max_splice_rms=0.7,
        min_radius=0.0,
    )
    if args.clash_check < args.dump_pdb * 100:
        args.clash_check = args.dump_pdb * 100

    pyrosetta.init('-mute all -beta')

    db = (
        BBlockDB(
            bakerdb_files=glob.glob('worms/data/*.json'),
            read_new_pdbs=True,
            verbosity=args.verbosity
        ),
        SpliceDB(),
    )

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
    crit = spec[1]

    touter = time()
    ot_graph, ot_rslt, ot_crit = outside_grow(spec, **kw)

    assert 0

    ntot = len(ot_rslt.idx)
    keys, hash_table = _make_hash_table(ot_graph, ot_rslt, binner)
    print(
        '  nresult outer', len(ot_rslt.idx), 'unique hashes', len(set(keys)),
        'rate',
        f'{int(ot_rslt.stats.total_samples[0] / (time() - touter)):,}/s'
    )

    tinner = time()
    in_graph, in_rslt = inside_grow(
        spec, binner=numba_binner, table=hash_table, **kw
    )
    tinner = time() - tinner
    print(
        '  nresults inner', len(in_rslt.idx),
        f'rate {int(in_rslt.stats.total_samples[0] / tinner):,}/s'
    )
    # for i in range(min(len(in_rslt.idx), 10)):
    # graph_dump_pdb('inner_%i_bb.pdb' % i, in_graph, in_rslt.idx[i],
    # in_rslt.pos[i], join=0) # yapf: disable

    from_seg = spec[1].from_seg
    graph = linear_gragh(
        spec[0], modbbs=singlebb((from_seg, ), 0), make_edges=False, **kw
    )

    print('whole:', spec[0])

    rslt = _merge_results(
        graph, crit, in_rslt, in_graph, ot_rslt, ot_graph, binner, hash_table
    )
    ntot = len(rslt.idx)
    tclash = time()
    # rslt = prune_clashing_results(
    # graph, crit, rslt, at_most=1000, thresh=3.0, **kw
    # )
    tclash = time() - tclash
    print(
        '  nresults', len(rslt.idx), 'withclash', ntot, 'clashrate',
        ntot / tclash
    )

    for irslt in range(min(20, len(rslt.idx))):
        idx = rslt.idx[irslt]
        pos = rslt.pos[irslt]
        i_ot_rslt = rslt.stats[irslt]

        graph_dump_pdb('outer_%03i.pdb' % irslt, ot_graph, ot_rslt.idx[i_ot_rslt],
            ot_rslt.pos[i_ot_rslt], join='bb') # yapf: disable

        graph_dump_pdb('whole_%03i_bb.pdb' % irslt, graph, idx, pos, join='bb')

        pose = make_pose_crit(
            bbdb, graph, crit, idx, pos, join=True, only_connected='auto'
        )
        pose.dump_pdb('whole_%03i.pdb' % irslt)

        # # for i, p in enumerate(pose.split_by_chain()):
        # # p.dump_pdb('whole_%03i_%02i.pdb' % (irslt, i))

        # pose = make_pose_crit(
        #     bbdb, ot_graph, ot_crit, ot_rslt.idx[i_ot_rslt], ot_rslt.pos[i_ot_rslt],
        #     only_connected=False )
        # pose.dump_pdb('outer_%i.pdb' % irslt)


def _merge_results(
        graph, crit, in_rslt, in_graph, ot_rslt, ot_graph, binner, hash_table
):
    n = len(in_rslt.idx)
    nv = len(graph.verts)
    merged = SearchResult(
        pos=np.empty((n, nv, 4, 4), dtype='f8'),
        idx=np.empty((n, nv), dtype='i4'),
        err=in_rslt.err,
        stats=np.empty(n, dtype='i4')
    )
    ok = np.ones(n, dtype=np.bool)
    for i_in_rslt in range(n):
        val = _get_hash_val(
            binner, hash_table, in_rslt.pos[i_in_rslt, -1], crit.nfold
        )
        i_ot_rslt = np.right_shift(val, 32)
        assert i_ot_rslt < len(ot_rslt.idx)
        iouter = ot_rslt.idx[i_ot_rslt, 0]
        iouter2 = ot_rslt.idx[i_ot_rslt, -1]
        iinner = in_rslt.idx[i_in_rslt, -1]
        vouter = ot_graph.verts[0]
        vinner = in_graph.verts[-1]
        ibb = vouter.ibblock[iouter]
        ires_in = vinner.ires[iinner, 0]
        ires_out = vouter.ires[iouter, 1]
        isite_in = vinner.isite[iinner, 0]
        isite_out = vouter.isite[iouter, 1]
        isite_out2 = ot_graph.verts[-1].isite[iouter2, 0]
        mrgv = graph.verts[crit.from_seg]
        assert max(mrgv.ibblock) == 0
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
        assert len(idx) == len(graph.verts)
        for ii, v in zip(idx, graph.verts):
            assert ii < v.len
        pos = np.concatenate((
            in_rslt.pos[i_in_rslt, :-1],
            in_rslt.pos[i_in_rslt, -1] @ ot_rslt.pos[i_ot_rslt, :]
        ))
        assert np.allclose(pos[crit.from_seg], in_rslt.pos[i_in_rslt, -1])
        assert len(pos) == len(idx) == nv
        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.stats[i_in_rslt] = i_ot_rslt
    nbad = np.sum(1-ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    merged = SearchResult(
        pos=merged.pos[ok],
        idx=merged.idx[ok],
        err=merged.err[ok],
        stats=merged.stats[ok]
    )
    return merged


def singlebb(iverts, ibb):
    def func(bbs):
        for i in iverts:
            bbs[i] = (bbs[i][ibb], )

    return func


def ____outside_grow(spec, min_radius, **kw):
    bbty0, crit0 = spec
    bbty = deepcopy(bbty0[crit0.from_seg:])
    bbty[0][1] = '_' + bbty[0][1][1]
    print('outside', bbty)
    crit = Cyclic(crit0.nfold, min_radius=min_radius)
    graph = linear_gragh(bbty, modbbs=singlebb((0, -1), 0), **kw)

    if os.path.exists('outer.pickle'):
        with open('outer.pickle', 'rb') as inp:
            rslt = pickle.load(inp)
    else:
        rslt = grow_linear(
            graph,
            loss_function=crit.jit_lossfunc(),
            loss_threshold=1.0,
            last_bb_same_as=crit.from_seg,
            **kw
        )
        with open('outer.pickle', 'wb') as out:
            pickle.dump(rslt, out)

    return graph, rslt, crit


def outside_grow(spec, min_radius, **kw):
    bbty0, crit0 = spec
    bbty = deepcopy(bbty0[crit0.from_seg:])
    bbty[0][1] = '_' + bbty[0][1][1]
    print('outside', bbty)

    # remove this!
    # with open('outer_rslt.pickle', 'rb') as inp:
    #     rslt = pickle.load(inp)
    # crit = Cyclic(crit0.nfold, min_radius=min_radius)
    # graph = linear_gragh(bbty, modbbs=singlebb((0, -1), 0), **kw)
    # with open('outer.pickle', 'wb') as out:
    #     pickle.dump((graph, rslt, crit), out)

    if os.path.exists('outer.pickle'):
        with open('outer.pickle', 'rb') as inp:
            graph, rslt, crit = pickle.load(inp)
    else:
        crit = Cyclic(crit0.nfold, min_radius=min_radius)
        graph = linear_gragh(bbty, modbbs=singlebb((0, -1), 0), **kw)
        rslt = grow_linear(
            graph,
            loss_function=crit.jit_lossfunc(),
            loss_threshold=1.0,
            last_bb_same_as=crit.from_seg,
            **kw
        )
        with open('outer.pickle', 'wb') as out:
            pickle.dump((graph, rslt, crit), out)

    return graph, rslt, crit


def inside_grow(spec, binner, table, **kw):
    bbty0, crit0 = spec
    bbty = deepcopy(bbty0[:crit0.from_seg + 1])
    bbty[-1][1] = bbty[-1][1][0] + '_'
    print('inside', bbty)

    if os.path.exists('inner.pickle'):
        with open('inner.pickle', 'rb') as inp:
            graph, rslt = pickle.load(inp)
    else:
        graph = linear_gragh(bbty, modbbs=singlebb((-1, ), 0), **kw)
        rslt = grow_linear(
            graph,
            loss_function=_hash_lossfunc(binner, table, crit0.nfold),
            # loss_function=lossfunc_rand_1_in(1000),
            loss_threshold=1.0,
            **kw
        )
        with open('inner.pickle', 'wb') as out:
            pickle.dump((graph, rslt), out)
    return graph, rslt


def _get_args(**kw):
    p = argparse.ArgumentParser()
    for k, v in kw.items():
        p.add_argument('--' + k, type=type(v), dest=k, default=v)
    args = p.parse_args()
    return args


def _make_hash_table(graph, rslt, binner):
    keys = binner(rslt.pos[:, -1])
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
