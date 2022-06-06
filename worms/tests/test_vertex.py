import numba as nb
import numba.types as nt
import numpy as np
import pytest
import worms

# import worms.vertex
# from worms.vertex import *

def main():
   db = worms.data.db_bblock_caching_v0(worms.data.data_dir)
   test_Vertex_NC(db)
   test_Vertex_CN(db)
   test_Vertex__C(db)
   test_Vertex_N_(db)
   test_Vertex__N(db)

def test_Vertex_NC(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "NC")
   assert v.len == 55
   assert v.x2exit.shape == (55, 4, 4)
   assert v.x2orig.shape == (55, 4, 4)
   assert v.inout.shape == (55, 2)
   assert v.ires.shape == (55, 2)
   assert v.isite.shape == (55, 2)
   assert v.ichain.shape == (55, 2)
   assert v.ibblock.shape == (55, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   # print(np.concatenate([np.arange(len(v.inout))[:, None], v.inout], axis=1))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

def test_Vertex_CN(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "CN")
   assert v.len == 55
   assert v.x2exit.shape == (55, 4, 4)
   assert v.x2orig.shape == (55, 4, 4)
   assert v.inout.shape == (55, 2)
   assert v.ires.shape == (55, 2)
   assert v.isite.shape == (55, 2)
   assert v.ichain.shape == (55, 2)
   assert v.ibblock.shape == (55, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

def test_Vertex__C(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "_C")
   assert v.len == 25
   assert v.x2exit.shape == (25, 4, 4)
   assert v.x2orig.shape == (25, 4, 4)
   assert np.all(v.x2orig == np.eye(4))
   assert v.inout.shape == (25, 2)
   assert v.ires.shape == (25, 2)
   assert v.isite.shape == (25, 2)
   assert v.ichain.shape == (25, 2)
   assert v.ibblock.shape == (25, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires[:, 0] == -1)
   assert np.all(v.ires[:, 1] >= 0)
   assert np.all(v.ichain[:, 0] == -1)
   assert np.all(v.ichain[:, 1] >= 0)
   assert np.all(v.isite[:, 0] == -1)
   assert np.all(v.isite[:, 1] >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

def test_Vertex_N_(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "N_")
   assert v.len == 18
   assert v.x2exit.shape == (18, 4, 4)
   assert v.x2orig.shape == (18, 4, 4)
   assert v.inout.shape == (18, 2)
   assert v.ires.shape == (18, 2)
   assert v.isite.shape == (18, 2)
   assert v.ichain.shape == (18, 2)
   assert v.ibblock.shape == (18, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires[:, 1] == -1)
   assert np.all(v.ires[:, 0] >= 0)
   assert np.all(v.ichain[:, 1] == -1)
   assert np.all(v.ichain[:, 0] >= 0)
   assert np.all(v.isite[:, 1] == -1)
   assert np.all(v.isite[:, 0] >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

def test_Vertex__N(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "_N")
   assert v.len == 18
   assert v.x2exit.shape == (18, 4, 4)
   assert v.x2orig.shape == (18, 4, 4)
   assert v.inout.shape == (18, 2)
   assert v.ires.shape == (18, 2)
   assert v.isite.shape == (18, 2)
   assert v.ichain.shape == (18, 2)
   assert v.ibblock.shape == (18, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires[:, 0] == -1)
   assert np.all(v.ires[:, 1] >= 0)
   assert np.all(v.ichain[:, 0] == -1)
   assert np.all(v.ichain[:, 1] >= 0)
   assert np.all(v.isite[:, 0] == -1)
   assert np.all(v.isite[:, 1] >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

def test_Vertex_C_(db_bblock_caching_v0):
   bbs = db_bblock_caching_v0.query("all")
   v = worms.vertex.Vertex(bbs, "N_")
   assert v.len == 18
   assert v.x2exit.shape == (18, 4, 4)
   assert v.x2orig.shape == (18, 4, 4)
   assert v.inout.shape == (18, 2)
   assert v.ires.shape == (18, 2)
   assert v.isite.shape == (18, 2)
   assert v.ichain.shape == (18, 2)
   assert v.ibblock.shape == (18, )
   assert v.dirn.shape == (2, )
   assert np.all(v.ires[:, 1] == -1)
   assert np.all(v.ires[:, 0] >= 0)
   assert np.all(v.ichain[:, 1] == -1)
   assert np.all(v.ichain[:, 0] >= 0)
   assert np.all(v.isite[:, 1] == -1)
   assert np.all(v.isite[:, 0] >= 0)
   assert np.all(v.ibblock >= 0)
   assert np.all(v.ibblock < len(bbs))
   for i in range(v.inbreaks.size - 1):
      vals = v.inout[v.inbreaks[i]:v.inbreaks[i + 1], 0]
      assert np.all(vals == i)

if __name__ == '__main__':
   main()