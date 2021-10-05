from collections import namedtuple
import numpy as np
from numpy.lib import index_tricks
from worms.util import jit, expand_array_if_needed

SearchStats = namedtuple(
   "SearchStats", ["total_samples", "n_last_bb_same_as", "n_redundant_results", 'best_score'])

def zero_search_stats():
   return SearchStats(np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8"),
                      np.ones(1, dtype="f8") * 9e9)

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
   return ResultJIT(results.pos[isel], results.idx[isel], results.err[isel], results.stats)

def subset_result(results, ok):
   return ResultJIT(
      idx=results.idx[ok],
      pos=results.pos[ok],
      err=results.err[ok],
      stats=results.stats,
   )

class ResultTable(dict):
   def __init__(self, other):
      self.table = dict()
      self.idx = other.idx
      self.pos = other.pos
      self.err = other.err
      self.stats = other.stats

   def add(self, name, val):
      self.table[name] = val

   def update(self, order):
      for k, v in self.table.items():
         self.table[k] = v[order]

   def close_without_stats(self, other):
      # print('idxtype', type(self.idx), type(other.idx))
      # print('postype', type(self.pos), type(other.pos))
      # print('errtype', type(self.err), type(other.err))
      # print('statstype', type(self.stats), type(other.stats))

      idxeq = np.allclose(self.idx, other.idx)
      poseq = np.allclose(self.pos, other.pos)
      erreq = np.allclose(self.err, other.err)
      # statseq = self.stats == other.stats

      # print('ResultTable.__eq__', idxeq, poseq, erreq, statseq)

      return idxeq and poseq and erreq

   # def __getattr__(self, k):
   # if k in ('idx','pos','err','stats'):
   # return self.table[k]

   def __getstate__(self):
      return self.idx, self.pos, self.err, self.stats

   def __setstate__(self, state):
      self.idx, self.pos, self.err, self.stats = state
