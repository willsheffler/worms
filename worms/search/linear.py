import os
import numpy as np
import numba as nb
import types
from worms.util import jit, InProcessExecutor
from worms.vertex import _Vertex
from worms.edge import _Edge
from random import random
import concurrent.futures as cf
from worms.search.result import SearchResult, expand_results


@jit
def null_lossfunc(pos):
    return 0.0


def lossfunc_rand_1_in(n):
    @jit
    def func(pos):
        return float(random() * float(n))

    return func


def grow_linear(
        verts, edges, loss_function=null_lossfunc, loss_threshold=1.0,
        parallel=0
):
    assert len(verts) > 1
    assert len(verts) == len(edges) + 1
    assert verts[0].dirn[0] == 2
    assert verts[-1].dirn[1] == 2
    for ivertex in range(len(verts) - 1):
        assert verts[ivertex].dirn[1] + verts[ivertex + 1].dirn[0] == 1

    # if isinstance(loss_function, types.FunctionType):
    #     if not 'NUMBA_DISABLE_JIT' in os.environ:
    #         loss_function = nb.njit(nogil=1, fastmath=1)

    exe = cf.ThreadPoolExecutor if parallel else InProcessExecutor
    # exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        batch_size, vert0_size = 10, verts[0].len
        verts_pickleable = [v._state for v in verts]
        edges_pickleable = [e._state for e in edges]
        for ivert in range(0, verts[0].len, batch_size):
            futures.append(
                pool.submit(
                    _grow_linear_start,
                    verts_pickleable=verts_pickleable,
                    edges_pickleable=edges_pickleable,
                    loss_function=loss_function,
                    loss_threshold=loss_threshold,
                    nresults=0,
                    isplice=0,
                    ivertex_range=(ivert, min(vert0_size, ivert + batch_size)),
                    splice_position=np.eye(4),
                    showprogress=0
                )
            )
        results = [f.result() for f in futures]

    result = SearchResult(
        positions=np.concatenate([r.positions for r in results]),
        indices=np.concatenate([r.indices for r in results]),
        losses=np.concatenate([r.losses for r in results]),
    )

    return result


def _grow_linear_start(verts_pickleable, edges_pickleable, **kwargs):
    verts = tuple([_Vertex(*vp) for vp in verts_pickleable])
    edges = tuple([_Edge(*ep) for ep in edges_pickleable])
    positions = np.empty(shape=(1024, len(verts), 4, 4), dtype=np.float64)
    indices = np.empty(shape=(1024, len(verts)), dtype=np.int32)
    losses = np.empty(shape=(1024, ), dtype=np.float32)
    result = SearchResult(positions=positions, indices=indices, losses=losses)
    nresult, result = _grow_linear_recurse(result, verts, edges, **kwargs)
    result = SearchResult(*(a[:nresult] for a in result))
    return result


@jit
def _grow_linear_recurse(
        result, verts, edges, loss_function, loss_threshold, nresults, isplice,
        ivertex_range, splice_position, showprogress
):
    """Takes a partially built 'worm' of length isplice and extends them by one based on ivertex_range

    Args:
        result (SearchResult): accumulated positions, indices, and scores
        verts (tuple(_Vertex)*N): Vertices in the linear 'graph', store entry/exit geometry
        edges (tuple(_Edge)*(N-1)): Edges in the linear 'graph', store allowed splices
        loss_function (jit function): Arbitrary loss function, must be numba-jitable
        loss_threshold (float): only worms with loss <= loss_threshold are put into result
        nresults (int): total number of accumulated results so far
        isplice (int): index of current out-vertex / edge / splice
        ivertex_range (tuple(int, int)): range of ivertex with allowed entry ienter
        splice_position (float64[:4,:4]): rigid body position of splice
        showprogress (boolean): print progress messages

    Returns:
        (int, SearchResult): accumulated positions, indices, and scores
    """
    current_vertex = verts[isplice]
    for ivertex in range(*ivertex_range):
        if showprogress and isplice == 0:
            if (ivertex + 1) % (ivertex_range[1] / showprogress) == 0:
                print(int(ivertex * showprogress / ivertex_range[1]))
        result.indices[nresults, isplice] = ivertex
        vertex_position = splice_position @ current_vertex.x2orig[ivertex]
        result.positions[nresults, isplice] = vertex_position
        if isplice == len(edges):
            loss = loss_function(result.positions[nresults])
            result.losses[nresults] = loss
            if loss <= loss_threshold:
                nresults += 1
                result = expand_results(result, nresults)
        else:
            next_vertex = verts[isplice + 1]
            next_splicepos = splice_position @ current_vertex.x2exit[ivertex]
            iexit = current_vertex.exit_index[ivertex]
            allowed_entries = edges[isplice].allowed_entries(iexit)
            for ienter in allowed_entries:
                next_ivertex_range = next_vertex.entry_range(ienter)
                nresults, result = _grow_linear_recurse(
                    result=result,
                    verts=verts,
                    edges=edges,
                    loss_function=loss_function,
                    loss_threshold=loss_threshold,
                    nresults=nresults,
                    isplice=isplice + 1,
                    ivertex_range=next_ivertex_range,
                    splice_position=next_splicepos,
                    showprogress=showprogress
                )
    return nresults, result
