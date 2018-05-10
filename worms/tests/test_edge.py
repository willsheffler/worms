from worms import Vertex
from worms.edge import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis


# @pytest.mark.xfail
def test_Edge(bbdb):
    bbs = bbdb.query('all')
    for pre in '_NC':
        for post in '_NC':
            for d in ('CN', 'NC'):
                dirn = pre + d + post
                print(dirn[:2], dirn[2:])
                u = Vertex(bbs, np.arange(len(bbs)), dirn[:2])
                v = Vertex(bbs, np.arange(len(bbs)), dirn[2:])
                e = Edge(u, bbs, v, bbs)


def test_Edge_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')

    u = Vertex(bbs, np.arange(len(bbs)), '_C')
    v = Vertex(bbs, np.arange(len(bbs)), 'N_')
    e = Edge(u, bbs, v, bbs)

    u = Vertex(bbs, np.arange(len(bbs)), '_N')
    v = Vertex(bbs, np.arange(len(bbs)), 'C_')
    e = Edge(u, bbs, v, bbs)
