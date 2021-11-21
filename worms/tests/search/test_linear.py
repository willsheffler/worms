from worms.search.linear import grow_linear
from worms.vertex import Vertex
from worms.edge import Edge
from worms.database import CachingBBlockDB
from worms.ssdag import SearchSpaceDAG
import pytest
import numpy as np
import os
from worms.tests import only_if_jit

def _print_splices(e):
   for i in range(e.len):
      s = e.allowed_entries(i)
      if len(s):
         print(i, s)

def _num_splices(e):
   return sum(len(e.allowed_entries(i)) for i in range(e.len))

def _expand_inout_indices(verts, indices):
   new = np.empty((len(indices), len(verts) * 2 - 2), dtype=indices.dtype)
   new[:, 0] = indices[:, 0]
   for i in range(1, len(verts) - 1):
      new[:, 2 * i - 1] = verts[i].inout[indices[:, i], 0]
      new[:, 2 * i - 0] = verts[i].inout[indices[:, i], 1]
   new[:, -1] = indices[:, -1]
   return new

def test_linear_search_two(db_caching_bblock_v0_fullsize_prots):
   bbs = db_caching_bblock_v0_fullsize_prots.query("all")
   u = Vertex(bbs, "_C")
   v = Vertex(bbs, "N_")
   verts = (u, v)
   kw = dict(
      splice_max_rms=0.7,
      splice_ncontact_cut=30,
      splice_clash_d2=4.0**2,  # ca only
      splice_contact_d2=8.0**2,
      splice_rms_range=6,
      splice_clash_contact_range=60,
      splice_clash_contact_by_helix=False,
   )
   edges = (Edge(u, bbs, v, bbs, **kw)[0], )

   assert np.all(u.inout[:, 1] == np.arange(u.len))
   assert np.all(v.inout[:, 0] == np.arange(v.len))

   ssdag = SearchSpaceDAG(None, (bbs, ) * 2, verts, edges)
   result = grow_linear(ssdag, no_duplicate_bases=False)
   assert np.allclose(result.pos[:, 0], np.eye(4))

   isort = np.lexsort((result.idx[:, 1], result.idx[:, 0]))
   sortidx = result.idx[isort, :]
   print(repr(sortidx))
   assert np.all(sortidx == [
      [0, 3],
      [0, 24],
      [0, 41],
      [0, 60],
      [1, 22],
      [1, 25],
      [16, 3],
      [16, 39],
      [16, 40],
      [16, 57],
      [16, 60],
      [17, 0],
      [17, 22],
      [17, 40],
      [17, 55],
      [17, 58],
      [18, 23],
      [18, 38],
      [18, 55],
      [18, 59],
      [19, 24],
      [19, 41],
      [19, 56],
      [19, 60],
      [20, 18],
      [20, 57],
      [21, 58],
      [22, 20],
      [22, 23],
      [22, 38],
      [22, 39],
      [22, 59],
      [22, 60],
      [23, 24],
      [23, 39],
      [23, 40],
      [23, 41],
      [23, 54],
      [23, 60],
   ])

@only_if_jit
def test_linear_search_three(db_caching_bblock_v0_fullsize_prots):
   bbs = db_caching_bblock_v0_fullsize_prots.query("all")
   u = Vertex(bbs, "_C")
   v = Vertex(bbs, "NC")
   w = Vertex(bbs, "N_")
   verts = (u, v, w)
   kw = dict(
      splice_max_rms=0.5,
      splice_ncontact_cut=30,
      splice_clash_d2=4.0**2,  # ca only
      splice_contact_d2=8.0**2,
      splice_rms_range=6,
      splice_clash_contact_range=60,
      splice_clash_contact_by_helix=False,
   )
   e = Edge(u, bbs, v, bbs, **kw)[0]
   f = Edge(v, bbs, w, bbs, **kw)[0]
   edges = (e, f)

   # print('------------- e ---------------')
   # _print_splices(e)
   # print('------------- f ---------------')
   # _print_splices(f)
   # print('------------- result ---------------')

   ssdag = SearchSpaceDAG(None, (bbs, ) * 3, verts, edges)
   result = grow_linear(ssdag, no_duplicate_bases=False)

   # from time import clock
   # t = clock()
   # for i in range(100):
   # grow_linear(verts, edges)
   # print('time 10', clock() - t)
   # assert 0

   assert np.allclose(result.pos[:, 0], np.eye(4))

   idx = _expand_inout_indices(verts, result.idx)
   isort = np.lexsort((idx[:, 3], idx[:, 2], idx[:, 1], idx[:, 0]))
   idx = idx[isort, :]
   assert len(idx) == _num_splices(e) * _num_splices(f)

   # np.set_printoptions(threshold=np.nan)
   # print(repr(idx))

   assert np.all(idx == [
      [0, 19, 0, 3],
      [0, 19, 0, 60],
      [0, 19, 16, 39],
      [0, 19, 17, 0],
      [0, 19, 17, 58],
      [0, 19, 18, 59],
      [0, 19, 22, 20],
      [0, 19, 22, 59],
      [0, 19, 23, 39],
      [0, 19, 23, 40],
      [0, 19, 23, 60],
      [17, 17, 0, 3],
      [17, 17, 0, 60],
      [17, 17, 16, 39],
      [17, 17, 17, 0],
      [17, 17, 17, 58],
      [17, 17, 18, 59],
      [17, 17, 22, 20],
      [17, 17, 22, 59],
      [17, 17, 23, 39],
      [17, 17, 23, 40],
      [17, 17, 23, 60],
      [18, 18, 0, 3],
      [18, 18, 0, 60],
      [18, 18, 16, 39],
      [18, 18, 17, 0],
      [18, 18, 17, 58],
      [18, 18, 18, 59],
      [18, 18, 22, 20],
      [18, 18, 22, 59],
      [18, 18, 23, 39],
      [18, 18, 23, 40],
      [18, 18, 23, 60],
      [22, 18, 0, 3],
      [22, 18, 0, 60],
      [22, 18, 16, 39],
      [22, 18, 17, 0],
      [22, 18, 17, 58],
      [22, 18, 18, 59],
      [22, 18, 22, 20],
      [22, 18, 22, 59],
      [22, 18, 23, 39],
      [22, 18, 23, 40],
      [22, 18, 23, 60],
      [23, 19, 0, 3],
      [23, 19, 0, 60],
      [23, 19, 16, 39],
      [23, 19, 17, 0],
      [23, 19, 17, 58],
      [23, 19, 18, 59],
      [23, 19, 22, 20],
      [23, 19, 22, 59],
      [23, 19, 23, 39],
      [23, 19, 23, 40],
      [23, 19, 23, 60],
   ])

if __name__ == "__main__":
   db_caching_bblock_v0_fullsize_prots = CachingBBlockDB(
      cachedirs=[str(".worms_pytest_cache")],
      dbfiles=[os.path.join("worms/data/test_fullsize_prots.json")],
      lazy=False,
      read_new_pdbs=True,
      nprocs=1,
   )

   test_linear_search_two(db_caching_bblock_v0_fullsize_prots)
