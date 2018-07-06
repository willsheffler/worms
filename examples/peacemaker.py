import logging
import sys
import glob
from time import clock, time

import numpy as np
import pytest
from xbin import XformBinner

from worms import linear_gragh, Cyclic, grow_linear
from worms.database import BBlockDB, SpliceDB
from worms.graph_pose import make_pose
from worms.filters.clash import prune_clashing_results

logging.getLogger().setLevel(99)


def cyclic_bin_data(
        spec,
        db,
        maxbb=10,
        parallel=1,
        verbosity=1,
        monte_carlo=0,
):
    bbty, crit = spec

    gragh = linear_gragh(
        bbty,
        db,
        maxbb=maxbb,
        verbosity=verbosity,
        parallel=parallel,
    )

    rslt = grow_linear(
        graph,
        loss_function=crit.jit_lossfunc(),
        parallel=parallel,
        loss_threshold=1.0,
        last_bb_same_as=crit.from_seg,
        monte_carlo=monte_carlo
    )

    print('nresult', len(rslt.idx))

    binner = XformBinner(cart_resl, ori_resl)
    x = rslt.pos[:, -1]

    return rslt, bins


def make_peace(
        spec,
        db,
        maxbb=10,
        parallel=1,
        verbosity=1,
        monte_carlo=0,
        clash_check=0,
        dump_pdb=0,
        cache_sync=0.001
):
    if clash_check < dump_pdb: clash_check = dump_pdb * 100
    bbty, crit = spec

    bbty0 = bbty[crit.from_seg:]
    bbty[0][1] = '_' + bbty[0][1][1]
    crit0 = Cyclic(crit.nfold)
    spec0 = bbty0, crit0

    rslt, htbl = cyclic_bin_data(
        spec0,
        db,
        maxbb=maxbb,
        parallel=parallel,
        verbosity=verbosity,
        monte_carlo=monte_carlo
    )


def main():

    import argparse
    import pyrosetta
    import glob
    pyrosetta.init('-mute all -beta')

    p = argparse.ArgumentParser()
    p.add_argument('--verbosity', type=int, dest='verbosity', default=2)
    p.add_argument('--parallel', type=int, dest='parallel', default=1)
    p.add_argument('--nbblocks', type=int, dest='nbblocks', default=4)
    p.add_argument('--dump_pdb', type=int, dest='dump_pdb', default=0)
    p.add_argument('--clash_check', type=int, dest='clash_check', default=0)
    p.add_argument('--cache_sync', type=float, dest='cache_sync', default=0.01)
    p.add_argument('--monte_carlo', type=int, dest='monte_carlo', default=0)
    args = p.parse_args()

    bbdb = BBlockDB(
        bakerdb_files=glob.glob('worms/data/*.json'),
        read_new_pdbs=True,
        verbosity=args.verbosity
    )
    spdb = SpliceDB()
    db = bbdb, spdb

    crit = Cyclic(3, from_seg=2)
    bbty = [
        ['C3_N', '_N'],
        ['Het:CC', 'CC'],
        ['Het:NNX', 'NN'],
        ['Het:CC', 'CC'],
        ['Het:NNX', 'N_'],
    ]
    spec = bbty, crit

    make_peace(
        spec,
        db,
        maxbb=args.nbblocks,
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
