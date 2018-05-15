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

    for i in range(1, 21):
        bbs = bbs0 * i

        t = clock()
        u = Vertex(bbs, np.arange(len(bbs)), '_C')
        v = Vertex(bbs, np.arange(len(bbs)), 'N_')
        tv = clock() - t

        t = clock()
        m = splice_metrics(u, bbs, v, bbs)
        tm = clock() - t
        n = np.prod(m.rms.shape)
        nt = n / (tm)
        print(
            f'timing splice_metrics {i:4}, {n:10,} {tm:7.3f} {nt:10.1f} {tv:7.3f}'
        )


if __name__ == '__main__':
    main()