from worms.search.linear import grow_linear
from worms import Vertex, Edge, BBlockDB, Graph
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
    edges = (Edge(u, bbs, v, bbs, max_splice_rms=1.1, rms_range=9), )

    assert np.all(u.inout[:, 1] == np.arange(u.len))
    assert np.all(v.inout[:, 0] == np.arange(v.len))

    graph = Graph((bbs, ) * 2, verts, edges)
    result = grow_linear(graph)
    assert np.allclose(result.pos[:, 0], np.eye(4))

    isort = np.lexsort((result.idx[:, 1], result.idx[:, 0]))
    sortidx = result.idx[isort, :]
    assert np.all(
        sortidx == [[3, 28], [4, 29], [12, 37], [13, 38], [14, 19], [14, 39],
                    [16, 22], [16, 63], [17, 0], [17, 1], [17, 21], [17, 62],
                    [17, 64], [18, 0], [18, 1], [18, 22], [18, 63], [18, 64],
                    [19, 0], [19, 1], [19, 22], [19, 60], [19, 63], [19, 64]]
    ) # yapf: disable


@only_if_jit
def test_linear_search_three(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'NC')
    w = Vertex(bbs, 'N_')
    verts = (u, v, w)
    e = Edge(u, bbs, v, bbs, max_splice_rms=1.1, rms_range=9)
    f = Edge(v, bbs, w, bbs, max_splice_rms=1.1, rms_range=9)
    edges = (e, f)

    # print('------------- e ---------------')
    # _print_splices(e)
    # print('------------- f ---------------')
    # _print_splices(f)
    # print('------------- result ---------------')

    graph = Graph((bbs, ) * 3, verts, edges)
    result = grow_linear(graph)

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

    print(repr(idx))

    assert np.all(
        idx ==
        [[16, 22, 3, 28], [16, 22, 4, 29], [16, 22, 12, 37], [16, 22, 13, 38],
         [16, 22, 14, 19], [16, 22, 14, 39], [16, 22, 16, 22],
         [16, 22, 16, 63], [16, 22, 17, 0], [16, 22, 17, 1], [16, 22, 17, 21],
         [16, 22, 17, 62], [16, 22, 17, 64], [16, 22, 18, 0], [16, 22, 18, 1],
         [16, 22, 18, 22], [16, 22, 18, 63], [16, 22, 18, 64], [16, 22, 19, 0],
         [16, 22, 19, 1], [16, 22, 19, 22], [16, 22, 19, 60], [16, 22, 19, 63],
         [16, 22, 19, 64], [17, 21, 3, 28], [17, 21, 4, 29], [17, 21, 12, 37],
         [17, 21, 13, 38], [17, 21, 14, 19], [17, 21, 14, 39],
         [17, 21, 16, 22], [17, 21, 16, 63], [17, 21, 17, 0], [17, 21, 17, 1],
         [17, 21, 17, 21], [17, 21, 17, 62], [17, 21, 17, 64], [17, 21, 18, 0],
         [17, 21, 18, 1], [17, 21, 18, 22], [17, 21, 18, 63], [17, 21, 18, 64],
         [17, 21, 19, 0], [17, 21, 19, 1], [17, 21, 19, 22], [17, 21, 19, 60],
         [17, 21, 19, 63], [17, 21, 19, 64], [17, 23, 3, 28], [17, 23, 4, 29],
         [17, 23, 12, 37], [17, 23, 13, 38], [17, 23, 14, 19],
         [17, 23, 14, 39], [17, 23, 16, 22], [17, 23, 16, 63], [17, 23, 17, 0],
         [17, 23, 17, 1], [17, 23, 17, 21], [17, 23, 17, 62], [17, 23, 17, 64],
         [17, 23, 18, 0], [17, 23, 18, 1], [17, 23, 18, 22], [17, 23, 18, 63],
         [17, 23, 18, 64], [17, 23, 19, 0], [17, 23, 19, 1], [17, 23, 19, 22],
         [17, 23, 19, 60], [17, 23, 19, 63], [17, 23, 19, 64], [18, 22, 3, 28],
         [18, 22, 4, 29], [18, 22, 12, 37], [18, 22, 13, 38], [18, 22, 14, 19],
         [18, 22, 14, 39], [18, 22, 16, 22], [18, 22, 16, 63], [18, 22, 17, 0],
         [18, 22, 17, 1], [18, 22, 17, 21], [18, 22, 17, 62], [18, 22, 17, 64],
         [18, 22, 18, 0], [18, 22, 18, 1], [18, 22, 18, 22], [18, 22, 18, 63],
         [18, 22, 18, 64], [18, 22, 19, 0], [18, 22, 19, 1], [18, 22, 19, 22],
         [18, 22, 19, 60], [18, 22, 19, 63], [18, 22, 19, 64], [18, 23, 3, 28],
         [18, 23, 4, 29], [18, 23, 12, 37], [18, 23, 13, 38], [18, 23, 14, 19],
         [18, 23, 14, 39], [18, 23, 16, 22], [18, 23, 16, 63], [18, 23, 17, 0],
         [18, 23, 17, 1], [18, 23, 17, 21], [18, 23, 17, 62], [18, 23, 17, 64],
         [18, 23, 18, 0], [18, 23, 18, 1], [18, 23, 18, 22], [18, 23, 18, 63],
         [18, 23, 18, 64], [18, 23, 19, 0], [18, 23, 19, 1], [18, 23, 19, 22],
         [18, 23, 19, 60], [18, 23, 19, 63], [18, 23, 19, 64], [19, 19, 3, 28],
         [19, 19, 4, 29], [19, 19, 12, 37], [19, 19, 13, 38], [19, 19, 14, 19],
         [19, 19, 14, 39], [19, 19, 16, 22], [19, 19, 16, 63], [19, 19, 17, 0],
         [19, 19, 17, 1], [19, 19, 17, 21], [19, 19, 17, 62], [19, 19, 17, 64],
         [19, 19, 18, 0], [19, 19, 18, 1], [19, 19, 18, 22], [19, 19, 18, 63],
         [19, 19, 18, 64], [19, 19, 19, 0], [19, 19, 19, 1], [19, 19, 19, 22],
         [19, 19, 19, 60], [19, 19, 19, 63], [19, 19, 19, 64], [19, 22, 3, 28],
         [19, 22, 4, 29], [19, 22, 12, 37], [19, 22, 13, 38], [19, 22, 14, 19],
         [19, 22, 14, 39], [19, 22, 16, 22], [19, 22, 16, 63], [19, 22, 17, 0],
         [19, 22, 17, 1], [19, 22, 17, 21], [19, 22, 17, 62], [19, 22, 17, 64],
         [19, 22, 18, 0], [19, 22, 18, 1], [19, 22, 18, 22], [19, 22, 18, 63],
         [19, 22, 18, 64], [19, 22, 19, 0], [19, 22, 19, 1], [19, 22, 19, 22],
         [19, 22, 19, 60], [19, 22, 19, 63], [19, 22, 19, 64], [19, 23, 3, 28],
         [19, 23, 4, 29], [19, 23, 12, 37], [19, 23, 13, 38], [19, 23, 14, 19],
         [19, 23, 14, 39], [19, 23, 16, 22], [19, 23, 16, 63], [19, 23, 17, 0],
         [19, 23, 17, 1], [19, 23, 17, 21], [19, 23, 17, 62], [19, 23, 17, 64],
         [19, 23, 18, 0], [19, 23, 18, 1], [19, 23, 18, 22], [19, 23, 18, 63],
         [19, 23, 18, 64], [19, 23, 19, 0], [19, 23, 19, 1], [19, 23, 19, 22],
         [19, 23, 19, 60], [19, 23, 19, 63], [19, 23, 19, 64]]
    )  # yapf: disable


if __name__ == '__main__':
    bbdb_fullsize_prots = BBlockDB(
        cachedir=str('.worms_pytest_cache'),
        bakerdb_files=[os.path.join('worms/data/test_fullsize_prots.json')],
        lazy=False,
        read_new_pdbs=True,
        nprocs=1,
    )

    test_linear_search_two(bbdb_fullsize_prots)
