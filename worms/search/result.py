from collections import namedtuple
import numpy as np
from worms.util import jit, expand_array_if_needed

SearchStats = namedtuple('SearchStats', ['total_samples', 'n_last_bb_same_as'])


def zero_search_stats():
    return SearchStats(np.zeros(1), np.zeros(1))


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
