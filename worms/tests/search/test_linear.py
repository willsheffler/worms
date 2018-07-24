from worms.search.linear import grow_linear
from worms import Vertex, Edge, BBlockDB, SearchSpaceDag
import pytest
import numpy as np
import os
from worms.tests import only_if_jit


def _print_splices(e):
    for i in range(e.len):
        s = e.allowed_entries(i)
        if len(s):
            print(i, s)


def _num_splices(e):
    return sum(len(e.allowed_entries(i)) for i in range(e.len))


def _expand_inout_indices(verts, indices):
    new = np.empty((len(indices), len(verts) * 2 - 2), dtype=indices.dtype)
    new[:, 0] = indices[:, 0]
    for i in range(1, len(verts) - 1):
        new[:, 2 * i - 1] = verts[i].inout[indices[:, i], 0]
        new[:, 2 * i - 0] = verts[i].inout[indices[:, i], 1]
    new[:, -1] = indices[:, -1]
    return new


def test_linear_search_two(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'N_')
    verts = (u, v)
    kw = dict(
        splice_max_rms=0.7,
        splice_ncontact_cut=30,
        splice_clash_d2=4.0**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_rms_range=6,
        splice_clash_contact_range=60
    )
    edges = (Edge(u, bbs, v, bbs, **kw), )

    assert np.all(u.inout[:, 1] == np.arange(u.len))
    assert np.all(v.inout[:, 0] == np.arange(v.len))

    ssdag = SearchSpaceDag(None, (bbs, ) * 2, verts, edges)
    result = grow_linear(ssdag)
    assert np.allclose(result.pos[:, 0], np.eye(4))

    isort = np.lexsort((result.idx[:, 1], result.idx[:, 0]))
    sortidx = result.idx[isort, :]
    print(repr(sortidx))
    assert np.all(
        sortidx == [[ 0, 24], [ 0, 25], [ 0, 43], [13, 39], [13, 41], [13, 64], [14, 18], [14, 25], [14, 40], [14, 42], [14, 58], [15,  1], [15, 41], [16,  2], [16, 20], [16, 23], [16, 38], [16, 39], [16, 41], [16, 42], [16, 57], [16, 63], [17, 20], [17, 21], [17, 24], [17, 39], [17, 40], [17, 42], [18,  0], [18, 22], [18, 25], [18, 38], [18, 40], [19,  1], [19, 23], [19, 39]]
    ) # yapf: disable


@only_if_jit
def test_linear_search_three(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'NC')
    w = Vertex(bbs, 'N_')
    verts = (u, v, w)
    kw = dict(
        splice_max_rms=0.5,
        splice_ncontact_cut=30,
        splice_clash_d2=4.0**2,  # ca only
        splice_contact_d2=8.0**2,
        splice_rms_range=6,
        splice_clash_contact_range=60
    )
    e = Edge(u, bbs, v, bbs, **kw)
    f = Edge(v, bbs, w, bbs, **kw)
    edges = (e, f)

    # print('------------- e ---------------')
    # _print_splices(e)
    # print('------------- f ---------------')
    # _print_splices(f)
    # print('------------- result ---------------')

    ssdag = SearchSpaceDag(None, (bbs, ) * 3, verts, edges)
    result = grow_linear(ssdag)

    # from time import clock
    # t = clock()
    # for i in range(100):
    # grow_linear(verts, edges)
    # print('time 10', clock() - t)
    # assert 0

    assert np.allclose(result.pos[:, 0], np.eye(4))

    idx = _expand_inout_indices(verts, result.idx)
    isort = np.lexsort((idx[:, 3], idx[:, 2], idx[:, 1], idx[:, 0]))
    idx = idx[isort, :]
    assert len(idx) == _num_splices(e) * _num_splices(f)

    np.set_printoptions(threshold=np.nan)
    print(repr(idx))

    assert np.all( idx ==
        [[14,  1,  0, 25], [14,  1, 14, 40], [14,  1, 14, 42], [14,  1, 16, 39],
         [14,  1, 17, 20], [14,  1, 17, 21], [14,  1, 17, 24], [14,  1, 17, 40],
         [14,  1, 17, 42], [14,  1, 18, 22], [14,  1, 18, 38], [14,  1, 18, 40],
         [14,  1, 19, 23], [14,  1, 19, 39], [17,  1,  0, 25], [17,  1, 14, 40],
         [17,  1, 14, 42], [17,  1, 16, 39], [17,  1, 17, 20], [17,  1, 17, 21],
         [17,  1, 17, 24], [17,  1, 17, 40], [17,  1, 17, 42], [17,  1, 18, 22],
         [17,  1, 18, 38], [17,  1, 18, 40], [17,  1, 19, 23], [17,  1, 19, 39]]
    )  # yapf: disable


if __name__ == '__main__':
    bbdb_fullsize_prots = BBlockDB(
        cachedirs=[str('.worms_pytest_cache')],
        dbfiles=[os.path.join('worms/data/test_fullsize_prots.json')],
        lazy=False,
        read_new_pdbs=True,
        nprocs=1,
    )

    test_linear_search_two(bbdb_fullsize_prots)
