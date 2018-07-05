from time import time
import concurrent.futures as cf
import numpy as np
from worms import Vertex, Edge


def _validate_bbs_verts(bbs, verts):
    assert len(bbs) == len(verts)
    for bb, vert in zip(bbs, verts):
        assert 0 <= np.min(vert.ibblock)
        assert np.max(vert.ibblock) < len(bb)


class Graph:
    def __init__(self, bbs, verts, edges):
        _validate_bbs_verts(bbs, verts)
        # print(type(bbs[0][0]))
        # print(type(verts[0]))
        # print(type(edges[0]))
        self.bbs = bbs
        self.verts = verts
        self.edges = edges


def linear_gragh(
        spec,
        bbdb,
        spdb,
        maxbb=100,
        shuf=False,
        min_seg_len=15,
        parallel=False,
        verbosity=0,
        timing=0,
        cache_sync=0.001,
):
    queries, directions = zip(*spec)
    if verbosity > 0:
        print('bblock queries', queries)
        print('directions', directions)
    tdb = time()
    bbmap = {
        q: bbdb.query(q, max_bblocks=maxbb, shuffle=shuf)
        for q in set(queries)
    }
    bbs = [bbmap[q] for q in queries]
    # for bb in bbs:
    # random.shuffle(bb)
    tdb = time() - tdb
    if verbosity > 0:
        print(
            f'bblock creation time {tdb:7.3f}', 'num bbs:',
            [len(x) for x in bbs]
        )

    tvertex = time()
    exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        for bb, dirn in zip(bbs, directions):
            futures.append(pool.submit(Vertex, bb, dirn, min_seg_len=15))
        verts = [f.result() for f in futures]
    # verts = [
    # Vertex(bb, dirn, min_seg_len=15, parallel=parallel)
    # for bb, dirn in zip(bbs, directions)
    # ]
    tvertex = time() - tvertex
    if verbosity > 0:
        print(
            f'vertex creation time {tvertex:7.3f}', 'num verts',
            [v.len for v in verts]
        )

    tedge = time()
    edges = [
        Edge(verts[i], bbs[i], verts[i + 1], bbs[i + 1],
                  parallel=parallel, verbosity=verbosity, splicedb=spdb, sync_to_disk_every=cache_sync)
        for i in range(len(verts) - 1)
    ] # yapf: disable
    tedge = time() - tedge

    if verbosity > 0:
        print(
            f'edge creation time {tedge:7.3f}', 'num splices',
            [e.total_allowed_splices() for e in edges], 'num exits',
            [e.len for e in edges]
        )

    spdb.sync_to_disk()

    toret = Graph(bbs, verts, edges)
    if timing:
        toret = toret, tdb, tvertex, tedge
    return toret
