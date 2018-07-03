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
    edges = (Edge(u, bbs, v, bbs, rms_cut=1.1), )

    assert np.all(u.inout[:, 1] == np.arange(u.len))
    assert np.all(v.inout[:, 0] == np.arange(v.len))

    graph = Graph((bbs, ) * 2, verts, edges)
    result = grow_linear(graph)
    assert np.allclose(result.pos[:, 0], np.eye(4))

    isort = np.lexsort((result.idx[:, 1], result.idx[:, 0]))
    sortidx = result.idx[isort, :]
    assert np.all(
        sortidx == [[0, 22], [18, 40], [19, 21], [19, 60], [21, 0],
                    [21, 58], [22, 1], [22, 57], [22, 59], [22, 60],
                    [23, 20], [23, 58], [23, 59], [23, 60]])  # yapf: disable

@only_if_jit
def test_linear_search_three(bbdb_fullsize_prots):
    bbs = bbdb_fullsize_prots.query('all')
    u = Vertex(bbs, '_C')
    v = Vertex(bbs, 'NC')
    w = Vertex(bbs, 'N_')
    verts = (u, v, w)
    e = Edge(u, bbs, v, bbs, rms_cut=1.1)
    f = Edge(v, bbs, w, bbs, rms_cut=1.1)
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

    assert np.all(idx == [
        [19, 19, 0, 22], [19, 19, 18, 40], [19, 19, 19, 21], [19, 19, 19, 60],
        [19, 19, 21, 0], [19, 19, 21, 58], [19, 19, 22, 1], [19, 19, 22, 57],
        [19, 19, 22, 59], [19, 19, 22, 60], [19, 19, 23, 20], [19, 19, 23, 58],
        [19, 19, 23, 59], [19, 19, 23, 60], [21, 17, 0, 22], [21, 17, 18, 40],
        [21, 17, 19, 21], [21, 17, 19, 60], [21, 17, 21, 0], [21, 17, 21, 58],
        [21, 17, 22, 1], [21, 17, 22, 57], [21, 17, 22, 59], [21, 17, 22, 60],
        [21, 17, 23, 20], [21, 17, 23, 58], [21, 17, 23, 59], [21, 17, 23, 60],
        [22, 16, 0, 22], [22, 16, 18, 40], [22, 16, 19, 21], [22, 16, 19, 60],
        [22, 16, 21, 0], [22, 16, 21, 58], [22, 16, 22, 1], [22, 16, 22, 57],
        [22, 16, 22, 59], [22, 16, 22, 60], [22, 16, 23, 20], [22, 16, 23, 58],
        [22, 16, 23, 59], [22, 16, 23, 60], [22, 18, 0, 22], [22, 18, 18, 40],
        [22, 18, 19, 21], [22, 18, 19, 60], [22, 18, 21, 0], [22, 18, 21, 58],
        [22, 18, 22, 1], [22, 18, 22, 57], [22, 18, 22, 59], [22, 18, 22, 60],
        [22, 18, 23, 20], [22, 18, 23, 58], [22, 18, 23, 59], [22, 18, 23, 60],
        [22, 19, 0, 22], [22, 19, 18, 40], [22, 19, 19, 21], [22, 19, 19, 60],
        [22, 19, 21, 0], [22, 19, 21, 58], [22, 19, 22, 1], [22, 19, 22, 57],
        [22, 19, 22, 59], [22, 19, 22, 60], [22, 19, 23, 20], [22, 19, 23, 58],
        [22, 19, 23, 59], [22, 19, 23, 60], [23, 17, 0, 22], [23, 17, 18, 40],
        [23, 17, 19, 21], [23, 17, 19, 60], [23, 17, 21, 0], [23, 17, 21, 58],
        [23, 17, 22, 1], [23, 17, 22, 57], [23, 17, 22, 59], [23, 17, 22, 60],
        [23, 17, 23, 20], [23, 17, 23, 58], [23, 17, 23, 59], [23, 17, 23, 60],
        [23, 18, 0, 22], [23, 18, 18, 40], [23, 18, 19, 21], [23, 18, 19, 60],
        [23, 18, 21, 0], [23, 18, 21, 58], [23, 18, 22, 1], [23, 18, 22, 57],
        [23, 18, 22, 59], [23, 18, 22, 60], [23, 18, 23, 20], [23, 18, 23, 58],
        [23, 18, 23, 59], [23, 18, 23, 60], [23, 19, 0, 22], [23, 19, 18, 40],
        [23, 19, 19, 21], [23, 19, 19, 60], [23, 19, 21, 0], [23, 19, 21, 58],
        [23, 19, 22, 1], [23, 19, 22, 57], [23, 19, 22, 59], [23, 19, 22, 60],
        [23, 19, 23, 20], [23, 19, 23, 58], [23, 19, 23, 59], [23, 19, 23, 60]
    ])  # yapf: disable


if __name__ == '__main__':
    bbdb_fullsize_prots = BBlockDB(
        cachedir=str('.worms_pytest_cache'),
        bakerdb_files=[os.path.join('worms/data/test_fullsize_prots.json')],
        lazy=False,
        read_new_pdbs=True,
        nprocs=1,
    )

    test_linear_search_two(bbdb_fullsize_prots)
