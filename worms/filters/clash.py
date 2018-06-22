import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count

from worms.util import jit, InProcessExecutor
from worms.search.result import SearchResult


@jit
def _get_all_chain_bounds(dirn, ires, idx, chains, trim=8):
    chains = np.copy(chains)
    if dirn[0] < 2:
        ir = ires[idx, 0]
        for i in range(len(chains)):
            lb, ub = chains[i]
            if lb <= ir < ub:
                chains[i, dirn[0]] = ir + trim * (1, -1)[dirn[0]]
                # chains[0], chains[i] = chains[i], chains[0]
    if dirn[1] < 2:
        ir = ires[idx, 1]
        for i in range(len(chains)):
            lb, ub = chains[i]
            if lb <= ir < ub:
                chains[i, dirn[1]] = ir + trim * (1, -1)[dirn[1]]
                # chains[0], chains[i] = chains[i], chains[0]
    return chains


@jit
def _get_trimmed_chain_bounds(dirn, ires, idx, chains, trim=8):
    chains = np.copy(chains)
    bounds = []
    if dirn[0] < 2:
        ir = ires[idx, 0]
        for i in range(len(chains)):
            lb, ub = chains[i]
            if lb <= ir < ub:
                chains[i, dirn[0]] = ir + trim * (1, -1)[dirn[0]]
                bounds.append((chains[i, 0], chains[i, 1]))
    if dirn[1] < 2:
        ir = ires[idx, 1]
        for i in range(len(chains)):
            lb, ub = chains[i]
            if lb <= ir < ub:
                chains[i, dirn[1]] = ir + trim * (1, -1)[dirn[1]]
                bounds.append((chains[i, 0], chains[i, 1]))
                # chains[0], chains[i] = chains[i], chains[0]
    return bounds


@jit
def _check_all_chain_clashes(
        dirns, iress, indices, position, chains, ncacs, thresh
):
    for i in range(len(dirns) - 1):
        ichntrm = _get_trimmed_chain_bounds(
            dirns[i], iress[i], indices[i], chains[i], 8
        )
        for j in range(i + 1, i + 2):
            jchntrm = _get_trimmed_chain_bounds(
                dirns[j], iress[j], indices[j], chains[j], 8
            )
            for ichain in range(len(ichntrm)):
                ilb, iub = ichntrm[ichain]
                for jchain in range(len(jchntrm)):
                    jlb, jub = jchntrm[jchain]
                    for ir in range(ilb, iub):
                        ica = position[i] @ ncacs[i][ir, 1]
                        for jr in range(jlb, jub):
                            jca = position[j] @ ncacs[j][jr, 1]
                            d2 = np.sum((ica - jca)**2)
                            if d2 < thresh:
                                return False
    for i in range(len(dirns) - 1):
        ichains = _get_all_chain_bounds(
            dirns[i], iress[i], indices[i], chains[i], 8
        )
        for j in range(i + 1, i + 2):
            jchains = _get_all_chain_bounds(
                dirns[j], iress[j], indices[j], chains[j], 8
            )
            for ichain in range(len(ichains)):
                ilb, iub = ichains[ichain]
                for jchain in range(len(jchains)):
                    jlb, jub = jchains[jchain]
                    for ir in range(ilb, iub):
                        ica = position[i] @ ncacs[i][ir, 1]
                        for jr in range(jlb, jub):
                            jca = position[j] @ ncacs[j][jr, 1]
                            d2 = np.sum((ica - jca)**2)
                            if d2 < thresh:
                                return False
    for i in range(len(dirns) - 1):
        ichains = _get_all_chain_bounds(
            dirns[i], iress[i], indices[i], chains[i], 8
        )
        for j in range(i + 1, len(dirns)):
            jchains = _get_all_chain_bounds(
                dirns[j], iress[j], indices[j], chains[j], 8
            )
            for ichain in range(len(ichains)):
                ilb, iub = ichains[ichain]
                for jchain in range(len(jchains)):
                    jlb, jub = jchains[jchain]
                    for ir in range(ilb, iub):
                        ica = position[i] @ ncacs[i][ir, 1]
                        for jr in range(jlb, jub):
                            jca = position[j] @ ncacs[j][jr, 1]
                            d2 = np.sum((ica - jca)**2)
                            if d2 < thresh:
                                return False
    return True


def prune_clashing_results(graph, worms, thresh=4.0, parallel=False):
    verts = tuple(graph.verts)

    exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        for i in range(len(worms.indices)):
            dirns = tuple([v.dirn for v in verts])
            iress = tuple([v.ires for v in verts])
            chains = tuple([
                graph.bbs[k][verts[k].ibblock[worms.indices[i, k]]].chains
                for k in range(len(graph.verts))
            ])
            ncacs = tuple([
                graph.bbs[k][verts[k].ibblock[worms.indices[i, k]]].ncac
                for k in range(len(graph.verts))
            ])
            futures.append(
                pool.submit(
                    _check_all_chain_clashes, dirns, iress, worms.indices[i],
                    worms.positions[i], chains, ncacs, thresh * thresh
                )
            )
        ok = np.array([f.result() for f in futures], dtype='?')

    return SearchResult(
        worms.positions[ok], worms.indices[ok], worms.losses[ok]
    )