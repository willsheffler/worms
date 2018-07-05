from worms import Vertex
from worms.tests import only_if_jit
from worms.edge import *
from worms.edge import _ires_from_conn
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis
import xarray as xr


@only_if_jit
def test_get_allowed_splices_fullsize_prots(bbdb_fullsize_prots):
    bbn, bbc = bbdb_fullsize_prots.query('all')

    iresc = _ires_from_conn(bbc.connections, 1)
    iresn = _ires_from_conn(bbn.connections, 0)
    assert list(iresc) == [
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34
    ]
    assert list(iresn) == [
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 344, 345, 346, 347, 348
    ]

    rms, nclash, ncontact = splice_metrics_pair(bbc, bbn, skip_on_fail=False)
    print(rms.shape)

    assert np.sum(rms < 1.5) == 30
    assert np.sum(rms < 1.1) == 16
    assert np.sum(rms < 0.7) == 0

    assert np.sum(nclash) == 575

    assert np.sum(ncontact) == 18149
    assert np.sum(ncontact > 9) == 796


@only_if_jit
def test_edge_fullsize_prots(bbdb_fullsize_prots, spdb):
    bbs = bbdb_fullsize_prots.query('all')
    # spdb = None

    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'N_')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(3) == [28])
    assert np.all(e.allowed_entries(4) == [29])
    assert np.all(e.allowed_entries(12) == [37])
    assert np.all(e.allowed_entries(13) == [38])
    assert np.all(e.allowed_entries(14) == [19, 39])
    assert np.all(e.allowed_entries(16) == [22, 63])
    assert np.all(e.allowed_entries(17) == [0, 1, 21, 62, 64])
    assert np.all(e.allowed_entries(18) == [0, 1, 22, 63, 64])
    assert np.all(e.allowed_entries(19) == [0, 1, 22, 60, 63, 64])
    for i in [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 15]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, 'NC')
    v = Vertex(bbs, 'NN')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(3) == [28])
    assert np.all(e.allowed_entries(4) == [29])
    assert np.all(e.allowed_entries(12) == [37])
    assert np.all(e.allowed_entries(13) == [38])
    assert np.all(e.allowed_entries(14) == [19, 39])
    assert np.all(e.allowed_entries(16) == [22])
    assert np.all(e.allowed_entries(17) == [0, 1, 21])
    assert np.all(e.allowed_entries(18) == [0, 1, 22])
    assert np.all(e.allowed_entries(19) == [0, 1, 22])
    for i in [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 15]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, 'NC')
    v = Vertex(bbs, 'NC')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(16) == [22])
    assert np.all(e.allowed_entries(17) == [21, 23])
    assert np.all(e.allowed_entries(18) == [22, 23])
    assert np.all(e.allowed_entries(19) == [19, 22, 23])
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, '_N')
    v = Vertex(bbs, 'CN')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(0) == [17, 18, 19])
    assert np.all(e.allowed_entries(1) == [17, 18, 19])
    assert np.all(e.allowed_entries(19) == [14])
    assert np.all(e.allowed_entries(21) == [17])
    assert np.all(e.allowed_entries(22) == [16, 18, 19])
    assert np.all(e.allowed_entries(28) == [3])
    assert np.all(e.allowed_entries(29) == [4])
    assert np.all(e.allowed_entries(37) == [12])
    assert np.all(e.allowed_entries(38) == [13])
    assert np.all(e.allowed_entries(39) == [14])
    assert np.all(e.allowed_entries(60) == [19])
    assert np.all(e.allowed_entries(62) == [17])
    assert np.all(e.allowed_entries(63) == [16, 18, 19])
    assert np.all(e.allowed_entries(64) == [17, 18, 19])
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
              23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43,
              44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
              61]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, '_N')
    v = Vertex(bbs, 'C_')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(0) == [17, 18, 19])
    assert np.all(e.allowed_entries(1) == [17, 18, 19])
    assert np.all(e.allowed_entries(19) == [14])
    assert np.all(e.allowed_entries(21) == [17])
    assert np.all(e.allowed_entries(22) == [16, 18, 19])
    assert np.all(e.allowed_entries(28) == [3])
    assert np.all(e.allowed_entries(29) == [4])
    assert np.all(e.allowed_entries(37) == [12])
    assert np.all(e.allowed_entries(38) == [13])
    assert np.all(e.allowed_entries(39) == [14])
    assert np.all(e.allowed_entries(60) == [19])
    assert np.all(e.allowed_entries(62) == [17])
    assert np.all(e.allowed_entries(63) == [16, 18, 19])
    assert np.all(e.allowed_entries(64) == [17, 18, 19])
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
              23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43,
              44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
              61]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, 'NN')
    v = Vertex(bbs, 'C_')
    e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    assert np.all(e.allowed_entries(0) == [17])
    assert np.all(e.allowed_entries(1) == [16, 18, 19])
    assert np.all(e.allowed_entries(7) == [3])
    assert np.all(e.allowed_entries(8) == [4])
    assert np.all(e.allowed_entries(16) == [12])
    assert np.all(e.allowed_entries(17) == [13])
    assert np.all(e.allowed_entries(18) == [14])
    assert np.all(e.allowed_entries(20) == [17, 18, 19])
    assert np.all(e.allowed_entries(21) == [17, 18, 19])
    assert np.all(e.allowed_entries(39) == [14])
    for i in [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 19, 22, 23, 24, 25, 26,
              27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]:
        assert len(e.allowed_entries(i)) == 0

    # #
    # #
    # # !!!!!!!!!!!!!!!!!!!!!!!!!
    # #
    # u = Vertex(bbs, 'NN')
    # v = Vertex(bbs, 'C_')
    # e = Edge(u, bbs, v, bbs, splicedb=spdb, rms_cut=1.1, ncontact_cut=10)
    # empty = []
    # for i in range(e.len):
    #     ent = list(e.allowed_entries(i))
    #     if len(ent):
    #         print('    assert np.all(e.allowed_entries(', i, ') == ', ent, ')')
    #     else:
    #         empty.append(i)
    # print('    for i in', empty, ':')
    # print('        assert len(e.allowed_entries(i)) == 0')
    # assert 0
