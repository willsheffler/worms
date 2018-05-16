import os
import numpy as np
import numba as nb
import types
from worms.util import jit, expand_array_if_needed
from collections import namedtuple
from worms.vertex import _Vertex
from worms.edge import _Edge


@jit
def null_lossfunc(pos):
    return 0.0


GrowResult = namedtuple('GrowResult', 'positions indices losses'.split())


@jit
def expand_results(result, nresults):
    if len(result[0]) <= nresults:
        result = GrowResult(
            expand_array_if_needed(result[0], nresults),
            expand_array_if_needed(result[1], nresults),
            expand_array_if_needed(result[2], nresults))
    result.indices[nresults] = result.indices[nresults - 1]
    result.positions[nresults] = result.positions[nresults - 1]
    return result


def grow_linear(vertex_chain,
                edges,
                loss_function=null_lossfunc,
                loss_threshold=1.0):
    assert len(vertex_chain) > 1
    assert len(vertex_chain) == len(edges) + 1
    assert vertex_chain[0].dirn[0] == 2
    assert vertex_chain[-1].dirn[1] == 2
    for ivertex in range(len(vertex_chain) - 1):
        assert vertex_chain[ivertex].dirn[1] + vertex_chain[ivertex
                                                            + 1].dirn[0] == 1

    if isinstance(loss_function, types.FunctionType):
        if not 'NUMBA_DISABLE_JIT' in os.environ:
            loss_function = nb.njit(nogil=1, fastmath=1)

    positions = np.empty(shape=(1, len(vertex_chain), 4, 4), dtype=np.float64)
    positions[:] = np.eye(4)
    indices = np.empty(shape=(1, len(vertex_chain)), dtype=np.int32)
    losses = np.empty(shape=(1, ), dtype=np.float32)
    result = GrowResult(positions=positions, indices=indices, losses=losses)

    nresult, result = _grow_linear_recurse(
        result=result,
        vertex_chain=vertex_chain,
        edges=edges,
        loss_function=loss_function,
        loss_threshold=loss_threshold,
        nresults=0,
        isplice=0,
        ivertex_range=(0, vertex_chain[0].len),
        splice_position=np.eye(4))

    result = GrowResult(*(a[:nresult] for a in result))

    return result


vert_type, edge_type = nb.deferred_type(), nb.deferred_type()
if not 'NUMBA_DISABLE_JIT' in os.environ:
    vert_type.define(_Vertex.class_type.instance_type)
    edge_type.define(_Edge.class_type.instance_type)


# @jitclass(('verts'))
class GrowGraphLinear:
    def __init__(self, verts, edges):
        self.verts = verts
        self.edges = edges


# @jitclass((''),)
class GrowStateLinear:
    def __init__(self):
        pass


class GrowAccumulator:
    def __init__(self):
        pass


# @jit
def _grow_linear_recurse(
        result,
        vertex_chain,
        edges,
        loss_function,
        loss_threshold,
        nresults,
        isplice,
        ivertex_range,
        splice_position,
):
    current_vertex = vertex_chain[isplice]
    for ivertex in range(*ivertex_range):
        result.indices[nresults, isplice] = ivertex
        vertex_position = splice_position @ current_vertex.x2orig[ivertex]
        result.positions[nresults, isplice] = vertex_position
        if isplice == len(edges):
            loss = loss_function(result.positions[nresults])
            result.losses[nresults] = loss
            if loss < loss_threshold:
                nresults += 1
                result = expand_results(result, nresults)
        else:
            next_vertex = vertex_chain[isplice + 1]
            next_splicepos = splice_position @ current_vertex.x2exit[ivertex]
            iexit = current_vertex.exit_index[ivertex]
            allowed_entries = edges[isplice].allowed_entries(iexit)
            for ienter in allowed_entries:
                next_ivertex_range = next_vertex.entry_range(ienter)
                nresults, result = _grow_linear_recurse(
                    result=result,
                    vertex_chain=vertex_chain,
                    edges=edges,
                    loss_function=loss_function,
                    loss_threshold=loss_threshold,
                    nresults=nresults,
                    isplice=isplice + 1,
                    ivertex_range=next_ivertex_range,
                    splice_position=next_splicepos,
                )

    return nresults, result
