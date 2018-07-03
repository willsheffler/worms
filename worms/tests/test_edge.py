from worms import Vertex
from worms.tests import only_if_jit
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
                u = Vertex(bbs, dirn[:2])
                v = Vertex(bbs, dirn[2:])
                m = splice_metrics(u, bbs, v, bbs)


@only_if_jit
def test_splice_metrics_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')

    ncontact_cut = 10
    rms_cut = 1.5

    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'N_')
    m = splice_metrics(u, bbs, v, bbs, skip_on_fail=False)

    nclash = np.sum(m.nclash == 0)
    ncontact = np.sum(m.ncontact >= ncontact_cut)
    nrms = np.sum(m.rms <= rms_cut)
    print(nrms, ncontact, nclash)
    assert nrms == 36
    assert nclash == 1213
    assert ncontact == 1419

    u = Vertex(bbs, '_N')
    v = Vertex(bbs, 'C_')
    m = splice_metrics(u, bbs, v, bbs, skip_on_fail=False)

    nclash = np.sum(m.nclash == 0)
    ncontact = np.sum(m.ncontact >= ncontact_cut)
    nrms = np.sum(m.rms <= rms_cut)
    assert nclash == 1213
    assert ncontact == 1419
    assert nrms == 36


@only_if_jit
def test_edge_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'N_')
    e = Edge(u, bbs, v, bbs, rms_cut=1.1, ncontact_cut=10)

    # print('allowed splices table')
    # print(e.splices.shape)
    # print(e.splices)
    # for i in range(e.len):
    # print(i, e.allowed_entries(i))
    assert np.all(e.allowed_entries(0) == [22])
    assert np.all(e.allowed_entries(1) == [])
    assert np.all(e.allowed_entries(2) == [])
    assert np.all(e.allowed_entries(3) == [])
    assert np.all(e.allowed_entries(4) == [])
    assert np.all(e.allowed_entries(5) == [])
    assert np.all(e.allowed_entries(6) == [])
    assert np.all(e.allowed_entries(7) == [])
    assert np.all(e.allowed_entries(8) == [])
    assert np.all(e.allowed_entries(9) == [])
    assert np.all(e.allowed_entries(10) == [])
    assert np.all(e.allowed_entries(11) == [])
    assert np.all(e.allowed_entries(12) == [])
    assert np.all(e.allowed_entries(13) == [])
    assert np.all(e.allowed_entries(14) == [])
    assert np.all(e.allowed_entries(15) == [])
    assert np.all(e.allowed_entries(16) == [])
    assert np.all(e.allowed_entries(17) == [])
    assert np.all(e.allowed_entries(18) == [40])
    assert np.all(e.allowed_entries(19) == [21, 60])
    assert np.all(e.allowed_entries(20) == [])
    assert np.all(e.allowed_entries(21) == [0, 58])
    assert np.all(e.allowed_entries(22) == [1, 57, 59, 60])
    assert np.all(e.allowed_entries(23) == [20, 58, 59, 60])
