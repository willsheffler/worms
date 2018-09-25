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
    # print(list(iresc))
    # print(list(iresn))
    assert list(iresc) == [
        537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550,
        551, 552, 553, 554, 555, 556, 557, 558, 559, 560
    ]
    assert list(iresn) == [
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 344, 345, 346, 347, 348
    ]

    rms, nclash, ncontact, ncnh, nhc = splice_metrics_pair(
        bbc,
        bbn,
        skip_on_fail=False,
        splice_rms_range=9,
        splice_max_rms=0.7,
        splice_clash_d2=3.0**2,
        splice_contact_d2=10.0**2,
        splice_clash_contact_range=9,
        splice_clash_contact_by_helix=False,
    )
    print(rms.shape)

    assert np.sum(rms < 1.5) == 18
    assert np.sum(rms < 1.1) == 6
    assert np.sum(rms < 0.7) == 0

    assert np.sum(nclash) == 80

    assert np.sum(ncontact) == 20711
    assert np.sum(ncontact > 9) == 944


@only_if_jit
def test_edge_fullsize_prots(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    # spdb = None

    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'N_')
    e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
        splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9, splice_clash_contact_by_helix=False
    ) # yapf: disable
    assert np.all(e.allowed_entries(0) == [1, 4, 26])
    assert np.all(e.allowed_entries(1) == [4, 26])
    assert np.all(e.allowed_entries(2) == [4, 5, 26, 42])
    assert np.all(e.allowed_entries(3) == [5, 42])
    assert np.all(e.allowed_entries(15) == [4])
    assert np.all(e.allowed_entries(16) == [26, 42])
    assert np.all(e.allowed_entries(17) == [4, 26, 42])
    assert np.all(e.allowed_entries(18) == [1, 26])
    assert np.all(e.allowed_entries(19) == [26, 42])
    assert np.all(e.allowed_entries(20) == [26])
    assert np.all(e.allowed_entries(21) == [4])
    assert np.all(e.allowed_entries(22) == [1])
    assert np.all(e.allowed_entries(23) == [26, 42])
    for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, 'NC')
    v = Vertex(bbs, 'NN')
    e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
        splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9,splice_clash_contact_by_helix=False
    ) # yapf: disable
    assert np.all(e.allowed_entries(0) == [1, 4, 26])
    assert np.all(e.allowed_entries(1) == [4, 26])
    assert np.all(e.allowed_entries(2) == [4, 5, 26])
    assert np.all(e.allowed_entries(3) == [5])
    assert np.all(e.allowed_entries(15) == [4])
    assert np.all(e.allowed_entries(16) == [26])
    assert np.all(e.allowed_entries(17) == [4, 26])
    assert np.all(e.allowed_entries(18) == [1, 26])
    assert np.all(e.allowed_entries(19) == [26])
    assert np.all(e.allowed_entries(20) == [26])
    assert np.all(e.allowed_entries(21) == [4])
    assert np.all(e.allowed_entries(22) == [1])
    assert np.all(e.allowed_entries(23) == [26])
    for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, '_N')
    v = Vertex(bbs, 'CN')
    e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
        splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9,splice_clash_contact_by_helix=False
    ) # yapf: disable
    assert np.all(e.allowed_entries(1) == [0, 18, 22])
    assert np.all(e.allowed_entries(4) == [0, 1, 2, 15, 17, 21])
    assert np.all(e.allowed_entries(5) == [2, 3])
    assert np.all(e.allowed_entries(26) == [0, 1, 2, 16, 17, 18, 19, 20, 23])
    assert np.all(e.allowed_entries(42) == [2, 3, 16, 17, 19, 23])
    for i in [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
              38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
              55, 56, 57, 58, 59, 60]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, '_N')
    v = Vertex(bbs, 'C_')
    e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
        splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9,splice_clash_contact_by_helix=False
    ) # yapf: disable
    assert np.all(e.allowed_entries(1) == [0, 18, 22])
    assert np.all(e.allowed_entries(4) == [0, 1, 2, 15, 17, 21])
    assert np.all(e.allowed_entries(5) == [2, 3])
    assert np.all(e.allowed_entries(26) == [0, 1, 2, 16, 17, 18, 19, 20, 23])
    assert np.all(e.allowed_entries(42) == [2, 3, 16, 17, 19, 23])
    for i in [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
              38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
              55, 56, 57, 58, 59, 60]:
        assert len(e.allowed_entries(i)) == 0

    u = Vertex(bbs, 'NN')
    v = Vertex(bbs, 'C_')
    e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
        splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9,splice_clash_contact_by_helix=False
    ) # yapf: disable
    assert np.all(e.allowed_entries(5) == [0, 1, 2, 16, 17, 18, 19, 20, 23])
    assert np.all(e.allowed_entries(21) == [0, 18, 22])
    assert np.all(e.allowed_entries(24) == [0, 1, 2, 15, 17, 21])
    assert np.all(e.allowed_entries(25) == [2, 3])
    for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
              38, 39, 40]:
        assert len(e.allowed_entries(i)) == 0

    # u = Vertex(bbs, 'NN')
    # v = Vertex(bbs, 'C_')
    # e = Edge(u, bbs, v, bbs, splice_max_rms=0.7,
    #     splice_rms_range=5, splice_ncontact_cut=7, splice_clash_contact_range=9
    # ) # yapf: disable
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