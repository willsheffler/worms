from time import time
import concurrent.futures as cf
import numpy as np
from worms import Vertex, Edge
from worms.bblock import bblock_dump_pdb, _BBlock
from worms.vertex import _Vertex
from worms.edge import _Edge
from worms.util import InProcessExecutor
from pprint import pprint
from logging import info
import string


def _validate_bbs_verts(bbs, verts):
    assert len(bbs) == len(verts)
    for bb, vert in zip(bbs, verts):
        assert 0 <= np.min(vert.ibblock)
        assert np.max(vert.ibblock) < len(bb)


class Graph:
    def __init__(self, bbs, verts, edges):
        _validate_bbs_verts(bbs, verts)
        assert isinstance(bbs[0][0], _BBlock)
        assert isinstance(verts[0], _Vertex)
        assert isinstance(edges[0], _Edge)
        self.bbs = tuple(bbs)
        self.verts = tuple(verts)
        self.edges = tuple(edges)

    def __getstate__(self):
        return ([[x._state for x in bb] for bb in self.bbs],
                [x._state for x in self.verts],
                [x._state for x in self.edges])

    def __setstate__(self, state):
        self.bbs = tuple(tuple(_BBlock(*x) for x in bb) for bb in state[0])
        self.verts = tuple(_Vertex(*x) for x in state[1])
        self.edges = tuple(_Edge(*x) for x in state[2])
        assert len(self.bbs) == len(self.verts) == len(self.edges) + 1


def linear_gragh(
        bbty,
        db,
        nbblocks=100,
        shuf=False,
        min_seg_len=15,
        parallel=False,
        verbosity=0,
        timing=0,
        cache_sync=0.001,
        modbbs=None,
        make_edges=True,
        **kw
):

    bbdb, spdb = db
    queries, directions = zip(*bbty)
    info('bblock queries', queries)
    info('directions', directions)
    tdb = time()
    bbmap = {
        q: bbdb.query(q, max_bblocks=nbblocks, shuffle=shuf)
        for q in set(queries)
    }
    bbs = [bbmap[q] for q in queries]
    if modbbs: modbbs(bbs)

    tdb = time() - tdb
    info(f'bblock creation time {tdb:7.3f}', 'num bbs:', [len(x) for x in bbs])

    tvertex = time()
    exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        for bb, dirn in zip(bbs, directions):
            futures.append(pool.submit(Vertex, bb, dirn, min_seg_len=15))
        verts = [f.result() for f in futures]

    tvertex = time() - tvertex
    info(
        f'vertex creation time {tvertex:7.3f}', 'num verts',
        [v.len for v in verts]
    )

    edges = None
    if make_edges:
        tedge = time()
        edges = [
            Edge(verts[i], bbs[i], verts[i + 1], bbs[i + 1],
                      parallel=parallel, verbosity=verbosity, splicedb=spdb,    sync_to_disk_every=cache_sync)
            for i in range(len(verts) - 1)
        ] # yapf: disable
        tedge = time() - tedge
        if verbosity > 0:
            print_edge_summary(edges)
        info(
            f'edge creation time {tedge:7.3f}', 'num splices',
            [e.total_allowed_splices() for e in edges], 'num exits',
            [e.len for e in edges]
        )
        spdb.sync_to_disk()

    toret = Graph(bbs, verts, edges)
    if timing:
        toret = toret, tdb, tvertex, tedge
    return toret


def print_edge_summary(edges):
    print('  splice stats: ', end='')
    for e in edges:
        nsplices = e.total_allowed_splices()
        ntot = e.nout * e.nent
        print(f'({nsplices:,} {nsplices*100.0/ntot:5.2f}%)', end=' ')
    print()


def graph_dump_pdb(out, graph, idx, pos, join=True):
    close = False
    if isinstance(out, str):
        out = open(out, 'w')
        close = True
    assert len(idx) == len(pos)
    assert idx.ndim == 1
    assert pos.ndim == 3
    assert pos.shape[-2:] == (4, 4)
    chain, anum, rnum = 0, 1, 1
    for i, tup in enumerate(zip(graph.bbs, graph.verts, idx, pos)):
        bbs, vert, ivert, x = tup
        chain, anum, rnum = bblock_dump_pdb(
            out=out,
            bblock=bbs[vert.ibblock[ivert]],
            dirn=vert.dirn,
            splice=vert.ires[ivert],
            pos=x,
            chain=chain,
            anum=anum,
            rnum=rnum,
            join=join,
        )
    if close: out.close()