import numpy as np

from worms.util import jit
from worms.search.result import SearchResult


@jit
def _ires_or_end(vert, bb, idx, iinout):
    assert idx < vert.ires.shape[0]
    ires = int(vert.ires[idx, iinout])
    if ires < 0:
        assert vert.dirn[iinout] == 2
        assert idx < vert.ichain.shape[0]
        chain_bounds = bb.chains[vert.ichain[idx, 1 - iinout], ]
        other_dirn = vert.dirn[1 - iinout]  # other
        return chain_bounds[1 - other_dirn]
    return ires


@jit
def _get_trimmed_chain_bounds(vert, idx, bb):
    ilb = _ires_or_end(vert, bb, idx, 0)
    iub = _ires_or_end(vert, bb, idx, 1)
    if ilb > iub:
        iub, ilb = ilb, iub
    return ilb + 8, iub - 8


@jit
def _check_contiguous_chain_clashes(verts, indices, position, bbs):
    for i in range(len(verts)):
        ilb, iub = _get_trimmed_chain_bounds(verts[i], indices[i], bbs[i])
        for j in range(i + 1, len(verts)):
            jlb, jub = _get_trimmed_chain_bounds(verts[j], indices[j], bbs[j])
            print(i, ilb, iub, j, jlb, jub)
            for ir in range(ilb, iub + 1):
                for ia in range(3):
                    icoord = position[i] @ bbs[i].ncac[ir, ia]
                    for jr in range(jlb, jlb + 1):
                        for ja in range(3):
                            jcoord = position[j] @ bbs[j].ncac[jr, ja]
                            d2 = np.sum((icoord - jcoord)**2)
                            print(i, ir, j, jr, icoord[:3], jcoord[:3], d2)
                            return False
                            if d2 < 9.0:
                                return False
    return True


def prune_clashing_results(graph, worms):
    ok = np.zeros(len(worms.indices), dtype='?')
    verts = tuple(graph.verts)
    for i in range(len(worms.indices)):
        bbs = tuple([
            graph.bbs[k][verts[k].ibblock[worms.indices[i, k]]]
            for k in range(len(graph.verts))
        ])
        ok[i] = _check_contiguous_chain_clashes(
            verts, worms.indices[i], worms.positions[i], bbs
        )
    print(ok)
    return SearchResult(
        worms.positions[ok], worms.indices[ok], worms.losses[ok]
    )