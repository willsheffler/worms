import logging
import sys
import concurrent.futures as cf
from time import clock, time

import numpy as np
import pytest

from worms import linear_gragh, Cyclic, grow_linear, NullCriteria
from worms.util import InProcessExecutor
from worms.database import BBlockDB, SpliceDB
from worms.graph_pose import make_pose_crit, make_pose
from worms.graph import graph_dump_pdb
from worms.filters.clash import prune_clashing_results
from worms.search import lossfunc_rand_1_in

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


def _dump_pdb(i, **kw):
    pose = make_pose(**kw)
    pose.dump_pdb('test_%i.pdb' % i)


def worm_grow_3(
        bbdb,
        spdb,
        nbblocks=10,
        shuf=0,
        parallel=1,
        verbosity=1,
        monte_carlo=0,
        clash_check=0,
        dump_pdb=0,
        cache_sync=0.001
):
    if clash_check < dump_pdb: clash_check = dump_pdb * 100
    ttot = time()

    graph, tdb, tvertex, tedge = linear_gragh(
        [
            ('C3_N', '_N'),
            ('Het:NCy', 'CN'),
            ('Het:CCC', 'C_'),
            # ('Het:NN', 'NN'),
            # ('Het:CC', 'CC'),
            # ('Het:NNX', 'N_'),
        ],
        (bbdb, spdb),
        nbblocks=nbblocks,
        timing=True,
        verbosity=verbosity,
        parallel=parallel,
        cache_sync=cache_sync
    )

    # crit = Cyclic(3, from_seg=2, origin_seg=0)
    # crit = Cyclic(3)
    # last_bb_same_as = crit.from_seg
    crit = NullCriteria()
    lf = crit.jit_lossfunc()
    last_bb_same_as = -1

    tgrow = time()
    rslt = grow_linear(
        graph,
        # loss_function=lf,
        loss_function=lossfunc_rand_1_in(1000),
        parallel=parallel,
        loss_threshold=1.0,
        last_bb_same_as=last_bb_same_as,
        monte_carlo=monte_carlo
    )
    tgrow = time() - tgrow

    Nres = len(rslt.err)
    Ntot = np.prod([v.len for v in graph.verts])
    logtot = np.log10(Ntot)
    print(
        'frac last_bb_same_as',
        rslt.stats.n_last_bb_same_as[0] / rslt.stats.total_samples[0]
    )
    Nsparse = int(rslt.stats.total_samples[0])
    Nsparse_rate = int(Nsparse / tgrow)
    ttot = time() - ttot
    if len(rslt.idx) == 0: frac_redundant = 0
    else: frac_redundant = rslt.stats.n_redundant_results[0] / len(rslt.idx)
    print(
        f' worm_grow_3 {nbblocks:4} {ttot:7.1f}s {Nres:9,} logtot{logtot:4.1f} tv'
        f' {tvertex:7.1f}s te {tedge:7.1f}s tg {tgrow:7.1f}s {Nsparse:10,} {Nsparse_rate:7,}/s {frac_redundant:4.1f}'
    )
    if len(rslt.err):
        print(
            'err 0 25 50 75 100',
            np.percentile(rslt.err, (0, 25, 50, 75, 100))
        )
    sys.stdout.flush()

    if not clash_check: return

    tclash = time()
    norig = len(rslt.idx)
    rslt = prune_clashing_results(
        graph, crit, rslt, at_most=clash_check, thresh=4.0, parallel=parallel
    )
    print(
        'pruned clashes, %i of %i remain,' %
        (len(rslt.idx), min(clash_check, norig)), 'took',
        time() - tclash, 'seconds'
    )

    for i, idx in enumerate(rslt.idx):
        graph_dump_pdb(
            'graph_%i_nojoin.pdb' % i, graph, idx, rslt.pos[i], join=0
        )
        graph_dump_pdb('graph_%i.pdb' % i, graph, idx, rslt.pos[i])

    return

    if len(rslt.idx) > 0:
        tpdb = time()
        exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
        with exe(max_workers=3) as pool:
            futures = list()
            for i in range(min(dump_pdb, len(rslt.idx))):
                kw = dict(
                    bbdb=bbdb,
                    graph=graph,
                    # crit=crit,
                    i=i,
                    indices=rslt.idx[i],
                    positions=rslt.pos[i],
                    only_connected=False,
                )
                futures.append(pool.submit(_dump_pdb, **kw))
            [f.result() for f in futures]
        print(
            'dumped %i structures' % min(dump_pdb, len(rslt.idx)), 'time',
            time() - tpdb
        )


def main():
    import argparse
    import glob
    import pyrosetta
    pyrosetta.init('-mute all -beta')

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbosity', type=int, dest='verbosity', default=0)
    parser.add_argument('--parallel', type=int, dest='parallel', default=True)
    parser.add_argument('--nbblocks', type=int, dest='nbblocks', default=4)
    parser.add_argument(
        '--clash_check', type=int, dest='clash_check', default=0
    )
    parser.add_argument('--dump_pdb', type=int, dest='dump_pdb', default=0)
    parser.add_argument(
        '--cache_sync', type=float, dest='cache_sync', default=0.01
    )
    parser.add_argument(
        '--monte_carlo', type=int, dest='monte_carlo', default=0
    )
    args = parser.parse_args()

    bbdb = BBlockDB(
        bakerdb_files=[
            'worms/data/c6_database.json',
            'worms/data/HBRP_Cx_database.json',
            'worms/data/HFuse_Cx_database.20180219.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.json',
            'worms/data/HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh13_3.20180406.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh13_3.20180416.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh29_3.20180406.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh29_3.20180416.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh34_3.20180416.json',
            'worms/data/HFuse_het_3chain_2arm_database.Sh3e_3.20180406.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh13_3.20180406.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh13_3.20180416.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh29_3.20180406.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh29_3.20180416.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh34_3.20180416.json',
            'worms/data/HFuse_het_3chain_3arm_database.Sh3e_3.20180406.json',
            'worms/data/master_database_generation2.json',
            'worms/data/test_db_file.json',
            'worms/data/test_fullsize_prots.json',
        ],
        read_new_pdbs=True,
        verbosity=args.verbosity
    )

    spdb = SpliceDB()

    worm_grow_3(
        bbdb,
        spdb,
        nbblocks=args.nbblocks,
        parallel=args.parallel,
        verbosity=args.verbosity,
        monte_carlo=args.monte_carlo,
        clash_check=args.clash_check,
        dump_pdb=args.dump_pdb,
        cache_sync=args.cache_sync,
    )
    sys.stdout.flush()


if __name__ == '__main__':
    main()
