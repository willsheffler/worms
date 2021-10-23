"""TODO: Summary
"""
import sys, os, re, itertools, operator, datetime
import numba, numpy as np
from hashlib import sha1

jit = numba.njit(nogil=True, fastmath=True)

def helix_range(ss):
   helixof = np.zeros_like(ss, dtype=np.int32) - 1
   nhelix = 0
   prevh = 0
   # 72 is 'H'
   for i in range(len(ss)):
      if ss[i] == 72 and prevh == 0:
         nhelix += 1
      prevh = ss[i] == 72
   hrange = np.zeros((nhelix, 2), dtype=np.int32)
   nhelix = 0
   for i in range(len(ss)):
      if ss[i] == 72 and prevh == 0:  # starth
         hrange[nhelix, 0] = i
      elif ss[i] != 72 and prevh == 1:  # endh
         hrange[nhelix, 1] = i
         nhelix += 1
      if ss[i] == 72:
         helixof[i] = nhelix
      prevh = ss[i] == 72
   return hrange, helixof

def datetimetag():
   now = datetime.datetime.now()
   return now.strftime('%Y_%m_%d_%H_%M_%S')

def seconds_between_datetimetags(tag1, tag2):
   t1 = datetime_from_tag(tag1)
   t2 = datetime_from_tag(tag2)
   duration = t2 - t1
   return duration.total_seconds()

def datetime_from_tag(tag):
   vals = tag.split('_')
   assert len(vals) == 6
   vals = list(map(int, vals))
   # if this code is actually in service after 2099...
   # this failing assertion will be the least of our troubles
   # even worse if it's before I was born.... (WHS)
   assert 1979 < vals[0] < 2100
   assert 0 < vals[1] <= 12  # months
   assert 0 < vals[2] <= 31  # days
   assert 0 < vals[3] <= 60  # hour
   assert 0 < vals[4] <= 60  # minute
   assert 0 < vals[5] <= 60  # second
   return datetime.datetime(*vals)

def bigprod(iterable):
   return functools.reduce(operator.mul, iterable, 1)

# I dont remember what this was for
# class MultiRange:
#    def __init__(self, nside):
#       """TODO: Summary
#
#         Args:
#             nside (TYPE): Description
#         """
#       self.nside = np.array(nside, dtype="i")
#       self.psum = np.concatenate([np.cumprod(self.nside[1:][::-1])[::-1], [1]])
#       assert np.all(self.psum > 0)
#       assert bigprod(self.nside[1:]) < 2**63
#       self.len = bigprod(self.nside)
#
#    def __getitem__(self, idx):
#       """
#         """
#       if isinstance(idx, slice):
#          return (self[i] for i in range(self.len)[idx])
#       if idx >= self.len:
#          raise StopIteration
#       return tuple((idx // self.psum) % self.nside)
#
#    def __len__(self):
#       """
#         """
#       return self.len

def first_duplicate(segs):
   for i in range(len(segs) - 1, 0, -1):
      for j in range(i):
         if segs[i].spliceables == segs[j].spliceables:
            return j
   return None

def dicts_to_items(inp):
   if isinstance(inp, list):
      return [dicts_to_items(x) for x in inp]
   elif isinstance(inp, dict):
      return [(k, dicts_to_items(v)) for k, v in inp.items()]
   return inp

def items_to_dicts(inp):
   ''
   if isinstance(inp, list) and isinstance(inp[0], tuple) and len(inp[0]) == 2:
      return {k: items_to_dicts(v) for k, v in inp}
   elif isinstance(inp, list):
      return [items_to_dicts(x) for x in inp]
   return inp

def hash_str_to_int(s):
   if isinstance(s, str):
      s = s.encode()
   buf = sha1(s).digest()[:8]
   return int(abs(np.frombuffer(buf, dtype="i8")[0]))

def map_resis_to_asu(n, resis):
   return sorted(set((x - 1) % n + 1 for x in resis))
