import glob
import os
import logging
from time import clock, time
import sys
import concurrent.futures as cf
import numba as nb
import numba.types as nt
import numpy as np
import pytest

from worms import Vertex, Graph
from worms.criteria import Cyclic
from worms.search import grow_linear, lossfunc_rand_1_in
from worms.edge import *
from worms.database import BBlockDB, SpliceDB
from worms import vis
from worms.graph_pose import make_pose
from worms.filters.clash import prune_clashing_results

logging.getLogger().setLevel(99)

# David's Defaults
# --max_chunk_length 170
# --nres_from_termini 80
# --max_sample 1e11
# --min_chunk_length 100
# --use_class True
# --prefix %s_n%s
# --err_cutoff 9.0
# --max_chain_length 400
# --min_seg_len 15
# --cap_number_of_pdbs_per_segment 150
# --clash_cutoff 1.5
# --superimpose_rmsd 0.7
# --superimpose_length 9
# --Nproc_for_sympose 8
# --max_number_of_fusions_to_evaluate 10000
# --database_files %s" '%(base,nrun,base,base,nrun,config_file,base,nrun,DATABASES)


def hacktest_edge_creation(bbdb, spdb, parallel=0):
    bbsN = bbdb.query('Het:NNX', max_bblocks=2, shuffle=0)
    bbsC = bbdb.query('Het:CCX', max_bblocks=2, shuffle=0)
    verts = (
        Vertex(bbsN, '_N', min_seg_len=15, parallel=1),
        Vertex(bbsC, 'CC', min_seg_len=15, parallel=1),
        Vertex(bbsN, 'N_', min_seg_len=15, parallel=1)
    )
    t = time()
    e = Edge(verts[0], bbsN, verts[1], bbsC, splicedb=spdb, parallel=1)
    ttot = time() - t
    assert e.total_allowed_splices() == 1518
    assert np.all(
        e.allowed_entries(37) == [
            31, 33, 34, 74, 76, 77, 117, 119, 331, 333, 334, 374, 376, 377,
            417, 419, 461, 462, 486, 490, 512, 540, 562, 586
        ]
    )
    verts = (
        Vertex(bbsC, '_C', min_seg_len=15, parallel=1),
        Vertex(bbsN, 'NN', min_seg_len=15, parallel=1),
        Vertex(bbsC, 'C_', min_seg_len=15, parallel=1)
    )
    t = time()
    e = Edge(verts[0], bbsC, verts[1], bbsN, parallel=1)
    ttot += time() - t
    print('edge time', ttot)
    assert e.total_allowed_splices() == 1518
    assert np.all(
        e.allowed_entries(461) == [
            33, 35, 36, 37, 85, 86, 87, 88, 109, 151, 152, 195, 233, 253, 333,
            335, 336, 337, 385, 386, 387, 388, 409, 451, 452, 495, 509, 551,
            552, 595
        ]
    )
    print(len(spdb.cache))


def linear_gragh(
        spec,
        bbdb,
        spdb,
        maxbb=100,
        shuf=False,
        min_seg_len=15,
        parallel=False,
        verbosity=0,
        timing=0
):
    queries, directions = zip(*spec)
    if verbosity > 0:
        print('bblock queries', queries)
        print('directions', directions)
    tdb = time()
    bbmap = {
        q: bbdb.query(q, max_bblocks=maxbb, shuffle=shuf)
        for q in set(queries)
    }
    bbs = [bbmap[q] for q in queries]
    tdb = time() - tdb
    if verbosity > 0:
        print(
            f'bblock creation time {tdb:7.3f}', 'num bbs:',
            [len(x) for x in bbs]
        )

    tvertex = time()
    verts = [
        Vertex(bb, dirn, min_seg_len=15, parallel=parallel)
        for bb, dirn in zip(bbs, directions)
    ]
    tvertex = time() - tvertex
    if verbosity > 0:
        print(
            f'vertex creation time {tvertex:7.3f}', 'num verts',
            [v.len for v in verts]
        )

    tedge = time()
    edges = [
        Edge(verts[i], bbs[i], verts[i + 1], bbs[i + 1],
                  parallel=parallel, verbosity=1, splicedb=spdb)
        for i in range(len(verts) - 1)
    ] # yapf: disable
    tedge = time() - tedge

    if verbosity > 0:
        print(
            f'edge creation time {tedge:7.3f}', 'num splices',
            [e.total_allowed_splices() for e in edges], 'num exits',
            [e.len for e in edges]
        )

    spdb.sync_to_disk()

    toret = Graph(bbs, verts, edges)
    if timing:
        toret = toret, tdb, tvertex, tedge
    return toret


def _dump_pdb(bbdb, graph, spec, i, idx, pos):
    pose = make_pose(bbdb, graph, spec, idx, pos)
    pose.dump_pdb('test_%i.pdb' % i)


def perf_grow_3(bbdb, spdb, maxbb=10, shuf=0, parallel=1, verbosity=1):

    ttot = time()

    graph, tdb, tvertex, tedge = linear_gragh(
        [
            # ('C3_N'   , '_N'),
            # ('Het:CC' , 'CC'),
            ('Het:NNX', '_N'),
            ('Het:CC', 'CC'),
            ('Het:NNX', 'N_'),
        ],
        bbdb,
        spdb,
        maxbb=maxbb,
        timing=True,
        verbosity=verbosity,
        parallel=parallel
    )

    # loss = lossfunc_cyclic_rot(0, 2, 120)
    # loss = lossfunc_rand_1_in(1000)

    # spec = Cyclic(3, from_seg=2, origin_seg=0)
    spec = Cyclic(3)
    last_bb_same_as = spec.from_seg

    tgrow = time()
    wrm = grow_linear(
        graph,
        loss_function=spec.jit_lossfunc(),
        # loss_function=lossfunc_rand_1_in(1000),
        parallel=True,
        loss_threshold=1.0,
        last_bb_same_as=last_bb_same_as,
        monte_carlo=0
    )
    tgrow = time() - tgrow

    Nres = len(wrm.err)
    Ntot = np.prod([v.len for v in graph.verts])
    logtot = np.log10(Ntot)
    print(
        'frac last_bb_same_as',
        wrm.stats.n_last_bb_same_as[0] / wrm.stats.total_samples[0]
    )
    Nsparse = int(wrm.stats.total_samples[0])
    Nsparse_rate = int(Nsparse / tgrow)
    ttot = time() - ttot
    print(
        f' perf_grow_3 {maxbb:4} {ttot:7.1f}s {Nres:9,} logtot{logtot:4.1f} tv'
        f' {tvertex:7.1f}s te {tedge:7.1f}s tg {tgrow:7.1f}s {Nsparse:10,} {Nsparse_rate:7,}/s'
    )
    if len(wrm.err):
        print(
            'err 0 25 50 75 100', np.percentile(wrm.err, (0, 25, 50, 75, 100))
        )
    sys.stdout.flush()

    tclash = time()
    norig = len(wrm.idx)
    wrm = prune_clashing_results(graph, spec, wrm, 4.0, parallel=parallel)
    print(
        'pruned clashes, %i of %i remain,' % (len(wrm.idx), norig), 'took',
        time() - tclash, 'seconds'
    )

    if len(wrm.idx) > 0:
        tpdb = time()
        exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
        with exe(max_workers=3) as pool:
            futures = list()
            for i in range(len(wrm.idx)):
                args = _dump_pdb, bbdb, graph, spec, i, wrm.idx[i], wrm.pos[i]
                futures.append(pool.submit(*args))
            [f.result() for f in futures]
        print('dumped %i structures' % len(wrm.idx), 'time', time() - tpdb)


def main():
    import pyrosetta
    pyrosetta.init('-mute all -beta')
    bbdb = BBlockDB(bakerdb_files=glob.glob('worms/data/*.json'))
    spdb = SpliceDB()

    if sys.argv[1] == 'test':
        print('running hacktest_edge_creation')
        tstart = time()
        hacktest_edge_creation(bbdb, spdb)
        spdb.sync_to_disk()
        print('PASS', time() - tstart)
        return

    # for i in (16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512):
    # for i in (1, 2, 3, 4, 6, 8, 12, 16):  #, 24, 32, 48, 64):
    if len(sys.argv) > 1:
        sizes = sys.argv[1:]
    else:
        sizes = [2]
    for i in sizes:
        i = int(i)
        # for i in (16, 32, 48, 64):
        perf_grow_3(bbdb, spdb, maxbb=i)
        sys.stdout.flush()

    spdb.sync_to_disk()


if __name__ == '__main__':
    main()

# perf_grow_3  16    32.9s      3,240     179,712,000,000 tv  11.540 te   15.326 tg    5.058 4.744
# perf_grow_3  32    91.4s     30,987   1,603,584,000,000 tv  23.563 te   48.451 tg   19.312 4.714
# perf_grow_3  64   529.1s    511,385  13,492,224,000,000 tv  45.049 te  193.828 tg  290.207 4.421
# perf_grow_3  96  1385.8s  1,378,550  55,427,825,664,000 tv  92.058 te  521.253 tg  772.408 4.604
# perf_grow_3 128  3198.6s  3,518,962 152,343,340,646,400 tv 151.825 te 1089.234 tg 1957.422 4.636
# perf_grow_3 160  6032.9s  7,081,434 325,579,253,760,000 tv 214.022 te 1875.143 tg 3943.634 4.663
# perf_grow_3 192 10326.6s 12,996,062 593,564,649,062,400 tv 279.217 te 2861.581 tg 7185.690 4.660

# npdb   runtime          Nsparse                Ndense     v0.1 runtime
#   16      32.9        3,240,000       179,712,000,000     hours
#   32      91.4       30,987,000     1,603,584,000,000     days
#   64     529.1      511,385,000    13,492,224,000,000     weeks
#   96    1385.8    1,378,550,000    55,427,825,664,000     months
#  128    3198.6    3,518,962,000   152,343,340,646,400     years
#  160    6032.9    7,081,434,000   325,579,253,760,000     more years
#  192   10326.6   12,996,062,000   593,564,649,062,400     many more years
