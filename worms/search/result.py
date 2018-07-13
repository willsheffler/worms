from collections import namedtuple
import numpy as np
from worms.util import jit, expand_array_if_needed

SearchStats = namedtuple(
    'SearchStats',
    ['total_samples', 'n_last_bb_same_as', 'n_redundant_results']
)


def zero_search_stats():
    return SearchStats(np.zeros(1), np.zeros(1), np.zeros(1))


SearchResult = namedtuple('SearchResult', ['pos', 'idx', 'err', 'stats'])


@jit
def expand_results(result, nresults):
    if len(result[0]) <= nresults:
        result = SearchResult(
            expand_array_if_needed(result[0], nresults),
            expand_array_if_needed(result[1], nresults),
            expand_array_if_needed(result[2], nresults),
            result.stats,
        )
    result.idx[nresults] = result.idx[nresults - 1]
    result.pos[nresults] = result.pos[nresults - 1]
    result.err[nresults] = result.err[nresults - 1]
    return result


def remove_duplicate_results(results):
    h = np.array([hash(row.data.tobytes()) for row in results.idx])
    _, isel = np.unique(h, return_index=True)
    results.stats.n_redundant_results[0] = len(h)
    return SearchResult(
        results.pos[isel], results.idx[isel], results.err[isel], results.stats
    )


def subset_result(results, ok):
    return SearchResult(
        idx=results.idx[ok],
        pos=results.pos[ok],
        err=results.err[ok],
        stats=results.stats
    )