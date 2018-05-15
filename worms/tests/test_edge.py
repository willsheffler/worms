from worms import Vertex
from worms.edge import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis
import xarray as xr


def test_splice_metrics_run(bbdb):
    bbs = bbdb.query('all')
    for pre in '_NC':
        for post in '_NC':
            for d in ('CN', 'NC'):
                dirn = pre + d + post
                u = Vertex(bbs, np.arange(len(bbs)), dirn[:2])
                v = Vertex(bbs, np.arange(len(bbs)), dirn[2:])
                m = splice_metrics(u, bbs, v, bbs)


def test_splice_metrics_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')

    ncontact_cut = 10
    rms_cut = 1.5

    u = Vertex(bbs, np.arange(len(bbs)), '_C')
    v = Vertex(bbs, np.arange(len(bbs)), 'N_')
    m = splice_metrics(u, bbs, v, bbs, skip_on_fail=False)

    nclash = np.sum(m.nclash == 0)
    ncontact = np.sum(m.ncontact >= ncontact_cut)
    nrms = np.sum(m.rms <= rms_cut)
    print(nrms, ncontact, nclash)
    assert nrms == 36
    assert nclash == 1213
    assert ncontact == 1419

    u = Vertex(bbs, np.arange(len(bbs)), '_N')
    v = Vertex(bbs, np.arange(len(bbs)), 'C_')
    m = splice_metrics(u, bbs, v, bbs, skip_on_fail=False)

    nclash = np.sum(m.nclash == 0)
    ncontact = np.sum(m.ncontact >= ncontact_cut)
    nrms = np.sum(m.rms <= rms_cut)
    assert nclash == 1213
    assert ncontact == 1419
    assert nrms == 36


def test_edge_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, np.arange(len(bbs)), '_C')
    v = Vertex(bbs, np.arange(len(bbs)), 'N_')
    e = Edge(u, bbs, v, bbs)

    # print('allowed splices table')
    # print(e.splices.shape)
    # print(e.splices)
    # for i in range(e.len):
    # print(i, e.allowed_splices(i))
    assert np.all(e.allowed_splices(0) == [22])
    assert np.all(e.allowed_splices(1) == [])
    assert np.all(e.allowed_splices(2) == [])
    assert np.all(e.allowed_splices(3) == [])
    assert np.all(e.allowed_splices(4) == [])
    assert np.all(e.allowed_splices(5) == [])
    assert np.all(e.allowed_splices(6) == [])
    assert np.all(e.allowed_splices(7) == [])
    assert np.all(e.allowed_splices(8) == [])
    assert np.all(e.allowed_splices(9) == [])
    assert np.all(e.allowed_splices(10) == [])
    assert np.all(e.allowed_splices(11) == [])
    assert np.all(e.allowed_splices(12) == [])
    assert np.all(e.allowed_splices(13) == [])
    assert np.all(e.allowed_splices(14) == [])
    assert np.all(e.allowed_splices(15) == [])
    assert np.all(e.allowed_splices(16) == [])
    assert np.all(e.allowed_splices(17) == [])
    assert np.all(e.allowed_splices(18) == [40])
    assert np.all(e.allowed_splices(19) == [21, 60])
    assert np.all(e.allowed_splices(20) == [])
    assert np.all(e.allowed_splices(21) == [0, 58])
    assert np.all(e.allowed_splices(22) == [1, 57, 59, 60])
    assert np.all(e.allowed_splices(23) == [20, 58, 59, 60])
