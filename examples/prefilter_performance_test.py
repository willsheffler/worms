from worms import Vertex
from worms.edge import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis
import os
from worms.database import BBlockDB
import logging

logging.getLogger().setLevel(99)


def main():
    bbdb_fullsize_prots = BBlockDB(
        cachedir=str('.worms_pytest_cache'),
        bakerdb_files=[os.path.join('worms/data/test_fullsize_prots.json')],
        lazy=False,
        read_new_pdbs=True,
        nprocs=1,
    )

    bbs0 = bbdb_fullsize_prots.query('all')

    ncontact_cut = 10
    rms_cut = 0.7

    from time import clock

    bbs = bbs0[1:]
    u = Vertex(bbs, np.arange(len(bbs)), '_C')
    v = Vertex(bbs, np.arange(len(bbs)), 'N_')
    splice_metrics(u, bbs, v, bbs)

    for i in range(20, 21):
        bbs = bbs0 * i

        # t = clock()
        u = Vertex(bbs, np.arange(len(bbs)), '_C')
        v = Vertex(bbs, np.arange(len(bbs)), 'N_')
        # print('timing vertex creation', i, u.len + v.len, clock() - t)

        t = clock()
        m = splice_metrics(u, bbs, v, bbs)
        print('timing splice_metrics', i, np.prod(m.rms.shape), clock() - t)


if __name__ == '__main__':
    main()