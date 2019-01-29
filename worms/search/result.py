from collections import namedtuple
import numpy as np
from worms.util import jit, expand_array_if_needed

SearchStats = namedtuple(
    "SearchStats", ["total_samples", "n_last_bb_same_as", "n_redundant_results"]
)


def zero_search_stats():
    return SearchStats(
        np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8")
    )


ResultJIT = namedtuple("ResultJIT", ["pos", "idx", "err", "stats"])


@jit
def expand_results(result, nresults):
    if len(result[0]) <= nresults:
        result = ResultJIT(
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
    return ResultJIT(
        results.pos[isel], results.idx[isel], results.err[isel], results.stats
    )


def subset_result(results, ok):
    return ResultJIT(
        idx=results.idx[ok],
        pos=results.pos[ok],
        err=results.err[ok],
        stats=results.stats,
    )


class ResultTable:
    def __init__(self, other):
        self.table = dict()
        self.table["idx"] = other.idx
        self.table["pos"] = other.pos
        self.table["err"] = other.err
        self.stats = other.stats

    def add(self, name, val):
        self.table[name] = val

    def update(self, order):
        for k, v in self.table.items():
            self.table[k] = v[order]

    def __getattr__(self, k):
        return self.table[k]
