from collections import namedtuple
import numpy as np
import worms

SearchStats = namedtuple(
   "SearchStats", ["total_samples", "n_last_bb_same_as", "n_redundant_results", 'best_score'])

def zero_search_stats():
   return SearchStats(np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8"), np.zeros(1, dtype="i8"),
                      np.ones(1, dtype="f8") * 9e9)

ResultJIT = namedtuple("ResultJIT", ["pos", "idx", "err", "stats"])

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

class Result:
   pass

class ResultTable(worms.Bunch, Result):
   def __init__(self, other, ssdag=None):
      self.idx = other.idx
      self.pos = other.pos
      self.err = other.err
      self.stats = other.stats
      self.ssdag = ssdag

   def __len__(self):
      assert len(self.idx) == len(self.pos) == len(self.err)
      return len(self.idx)

   def add(self, name, val):
      self[name] = val

   def update(self, array_index):
      for k, v in self.items():
         # TODO this is kinda hacky... maybe back to self.table?
         if not isinstance(v, np.ndarray):
            continue
         assert len(array_index) == len(v)
         # print(k)
         # print(v)
         # print(array_index)
         self[k] = v[array_index]

   def sort_on_idx(self):
      order = np.lexsort(np.flip(self.idx.T, axis=0))
      self.idx = self.idx[order]
      self.pos = self.pos[order]
      self.err = self.err[order]

   def approx_equal(self, other):
      # print('idxtype', self.idx.shape, other.idx.shape)
      # print('postype', self.pos.shape, other.pos.shape)
      # print('errtype', self.err.shape, other.err.shape)
      # print('statstype', type(self.stats), type(other.stats))
      if self.idx.shape != other.idx.shape:
         return False

      idxeq = np.allclose(self.idx, other.idx)
      poseq = np.allclose(self.pos, other.pos, atol=1e-6)
      erreq = np.allclose(self.err, other.err, atol=1e-3)
      # statseq = self.stats == other.stats

      a = self.idx
      b = other.idx
      # print(a.shape, b.shape)
      # print(a)
      # print(b)

      # assert idxeq
      # assert poseq
      # assert erreq

      r = idxeq and poseq and erreq
      return r

   def __getstate__(self):
      return self.idx, self.pos, self.err, self.stats

   def __setstate__(self, state):
      self.idx, self.pos, self.err, self.stats = state
