from worms import Vertex
from worms.edge import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis
import xarray as xr


@pytest.mark.skip
def test_splice_metrics(bbdb):
    bbs = bbdb.query('all')
    for pre in '_NC':
        for post in '_NC':
            for d in ('CN', 'NC'):
                dirn = pre + d + post
                print(dirn[:2], dirn[2:])
                u = Vertex(bbs, np.arange(len(bbs)), dirn[:2])
                v = Vertex(bbs, np.arange(len(bbs)), dirn[2:])
                e = splice_metrics(u, bbs, v, bbs)


@pytest.mark.xfail
def test_splice_metrics_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')

    u = Vertex(bbs, np.arange(len(bbs)), '_C')
    v = Vertex(bbs, np.arange(len(bbs)), 'N_')
    scm = splice_metrics(u, bbs, v, bbs)
    print(scm.nclash)
    assert isinstance(scm, xr.DataSet)

    u = Vertex(bbs, np.arange(len(bbs)), '_N')
    v = Vertex(bbs, np.arange(len(bbs)), 'C_')
    e = splice_metrics(u, bbs, v, bbs)
