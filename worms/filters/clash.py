import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count

from worms.util import jit, InProcessExecutor
from worms.search.result import SearchResult


def prune_clashing_results(
        graph,
        crit,
        rslt,
        at_most=-1,
        thresh=4.0,
        parallel=False,
):
    print('todo: clash check should handle symmetry')
    at_most = min(at_most, len(rslt.idx))
    if at_most < 0: at_most = len(rslt.idx)
    verts = tuple(graph.verts)
    exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
    with exe() as pool:
        futures = list()
        for i in range(at_most):
            dirns = tuple([v.dirn for v in verts])
            iress = tuple([v.ires for v in verts])
            chains = tuple([
                graph.bbs[k][verts[k].ibblock[rslt.idx[i, k]]].chains
                for k in range(len(graph.verts))
            ])
            ncacs = tuple([
                graph.bbs[k][verts[k].ibblock[rslt.idx[i, k]]].ncac
                for k in range(len(graph.verts))
            ])
            futures.append(
                pool.submit(
                    _check_all_chain_clashes, dirns, iress, rslt.idx[i],
                    rslt.pos[i], chains, ncacs, thresh * thresh
                )
            )
        ok = np.array([f.result() for f in futures], dtype='?')
    return SearchResult(
        rslt.pos[:at_most][ok], rslt.idx[:at_most][ok], rslt.err[:at_most][ok],
        rslt.stats
    )


@jit
def _chain_bounds(dirn, ires, idx, chains, spliced_only=False, trim=8):
    "return bounds for only spliced chains, with spliced away sequence removed"
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
    if spliced_only:
        return np.array(bounds, dtype=np.int32)
    else:
        return chains


@jit
def _has_ca_clash(position, ncacs, i, ichntrm, j, jchntrm, thresh):
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
                        return True
    return False


@jit
def _check_all_chain_clashes(dirns, iress, idx, pos, chn, ncacs, thresh):

    # only adjacent verts, only spliced chains
    for i in range(len(dirns) - 1):
        ichntrm = _chain_bounds(dirns[i], iress[i], idx[i], chn[i], 1, 8)
        for j in range(i + 1, i + 2):
            jchntrm = _chain_bounds(dirns[j], iress[j], idx[j], chn[j], 1, 8)
            if _has_ca_clash(pos, ncacs, i, ichntrm, j, jchntrm, thresh):
                return False

    # only adjacent verts, all chains
    for i in range(len(dirns) - 1):
        ichntrm = _chain_bounds(dirns[i], iress[i], idx[i], chn[i], 0, 8)
        for j in range(i + 1, i + 2):
            jchntrm = _chain_bounds(dirns[j], iress[j], idx[j], chn[j], 0, 8)
            if _has_ca_clash(pos, ncacs, i, ichntrm, j, jchntrm, thresh):
                return False

    # all verts, all chains
    for i in range(len(dirns) - 1):
        ichntrm = _chain_bounds(dirns[i], iress[i], idx[i], chn[i], 0, 8)
        for j in range(i + 1, len(dirns)):
            jchntrm = _chain_bounds(dirns[j], iress[j], idx[j], chn[j], 0, 8)
            if _has_ca_clash(pos, ncacs, i, ichntrm, j, jchntrm, thresh):
                return False

    return True
