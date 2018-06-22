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
from worms.search import grow_linear, lossfunc_rand_1_in
from worms.edge import *
from worms.database import BBlockDB
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


def linear_NC_gragh(n, uwv, efg):
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'NC')
    w = Vertex(bbs, 'N_')
    V = (u, ) + ((v, ) * (n - 2)) + (w, )
    assert len(V) == n

    e = Edge(u, bbs, v, bbs)
    f = Edge(v, bbs, v, bbs)
    g = Edge(v, bbs, w, bbs)
    E = (e, ) + ((f, ) * (n - 3)) + (g, )
    if n == 2:
        E = (Edge(u, bbs, w, bbs), )

    print('foo')
    print(E)
    assert len(E) == n - 1

    return V, E


def perf_grow_2(bbdb, maxbb=10, shuf=0):
    ttot = time()

    tdb = time()
    bbs = dict(
        C3_N=bbdb.query('C3_N', max_bblocks=maxbb, shuffle=shuf),
        C3_C=bbdb.query('C3_C', max_bblocks=maxbb, shuffle=shuf)
    )
    tdb = time() - tdb

    tvertex = time()
    ubbs = bbs['C3_N']
    vbbs = bbs['C3_C']
    bbs = (ubbs, vbbs)
    u = Vertex(ubbs, '_N', min_seg_len=15, parallel=1)
    v = Vertex(vbbs, 'C_', min_seg_len=15, parallel=1)
    V = (u, v)
    tvertex = time() - tvertex

    tedge = time()
    E = [Edge(u, ubbs, v, vbbs, parallel=1)]
    print('e.total_allowed_splices()', e.total_allowed_splices())
    tedge = time() - tedge
    # print(f'edge creation time {tedge:7.3f} {e.len} {f.len}')

    tgrow = time()
    worms = grow_linear(V, E, loss_function=lossfunc_rand_1_in(1), parallel=1)
    tgrow = time() - tgrow
    Nres = len(worms.losses)
    Ntot = u.len * v.len
    ttot = time() - ttot
    factor = np.log10(Ntot / (Nres + 1)) - 3  # every 1000th
    print(
        f' perf_grow_2 {maxbb:4} {ttot:7.1f}s {Nres:12,} {Ntot:20,} tv'
        f' {tvertex:7.1f}s te {tedge:7.1f}s tg {tgrow:7.1f}s {factor:10.3f}'
    )


def _dump_pdb(bbdb, graph, i, indices, positions):
    pose = make_pose(bbdb, graph, indices, positions, only_connected=0)
    pose.dump_pdb('test_%i.pdb' % i)


def perf_grow_3(bbdb, maxbb=10, shuf=0, parallel=1):
    ttot = time()

    tdb = time()
    bbmap = dict(
        C3_N=bbdb.query('C3_N', max_bblocks=maxbb, shuffle=shuf),
        Het_CCX=bbdb.query('Het:CCX', max_bblocks=maxbb, shuffle=shuf),
    )
    tdb = time() - tdb

    tvertex = time()
    bbs = (
        bbmap['C3_N'],
        bbmap['Het_CCX'],
        bbmap['C3_N'],
    )
    V = (
        Vertex(bbs[0], '_N', min_seg_len=15, parallel=parallel),
        Vertex(bbs[1], 'CC', min_seg_len=15, parallel=parallel),
        Vertex(bbs[2], 'N_', min_seg_len=15, parallel=parallel),
    )
    tvertex = time() - tvertex

    tedge = time()
    E = [
        Edge(
            V[i], bbs[i], V[i + 1], bbs[i + 1], parallel=parallel, verbosity=1
        ) for i in range(len(V) - 1)
    ]
    tedge = time() - tedge
    # print(f'edge creation time {tedgne:7.3f} {e.len} {f.len}')

    tgrow = time()
    w = grow_linear(
        V, E, loss_function=lossfunc_rand_1_in(1), parallel=parallel
    )
    tgrow = time() - tgrow
    Nres = len(w.losses)
    Ntot = np.prod([v.len for v in V])
    ttot = time() - ttot
    factor = np.log10(Ntot / (Nres + 1)) - 3  # every 1000th
    print(
        f' perf_grow_3 {maxbb:4} {ttot:7.1f}s {Nres:12,} {Ntot:20,} tv'
        f' {tvertex:7.1f}s te {tedge:7.1f}s tg {tgrow:7.1f}s {factor:10.3f}'
    )
    sys.stdout.flush()

    graph = Graph(bbs, V, E)

    tclash = time()
    norig = len(w.indices)
    w = prune_clashing_results(graph, w, parallel=parallel)
    print(
        'pruned clashes, %i of %i remain,' % (len(w.indices), norig), 'took',
        time() - tclash, 'seconds'
    )

    if len(w.indices) > 0:
        tpdb = time()
        exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
        with exe(max_workers=3) as pool:
            futures = list()
            for i in range(len(w.indices)):
                futures.append(
                    pool.submit(
                        _dump_pdb, bbdb, graph, i, w.indices[i], w.positions[i]
                    )
                )
            [f.result() for f in futures]
        print('dumped %i structures' % len(w.indices), 'time', time() - tpdb)


if __name__ == '__main__':
    import pyrosetta
    pyrosetta.init('-mute all -beta')
    bbdb = BBlockDB(bakerdb_files=glob.glob('worms/data/*.json'))
    # for i in (16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512):
    # for i in (1, 2, 3, 4, 6, 8, 12, 16):  #, 24, 32, 48, 64):
    if len(sys.argv) > 1:
        sizes = sys.argv[1:]
    else:
        sizes = [2]
    for i in sizes:
        i = int(i)
        # for i in (16, 32, 48, 64):
        perf_grow_3(bbdb, maxbb=i)
        sys.stdout.flush()

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
