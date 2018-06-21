from collections import namedtuple

from worms.util import jit, expand_array_if_needed

SearchResult = namedtuple('SearchResult', 'positions indices losses'.split())


@jit
def expand_results(result, nresults):
    if len(result[0]) <= nresults:
        result = SearchResult(
            expand_array_if_needed(result[0], nresults),
            expand_array_if_needed(result[1], nresults),
            expand_array_if_needed(result[2], nresults)
        )
    result.indices[nresults] = result.indices[nresults - 1]
    result.positions[nresults] = result.positions[nresults - 1]
    return result
