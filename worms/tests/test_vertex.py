from worms.vertex import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis

def test_Vertex_NC(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "NC")
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

def test_Vertex_CN(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "CN")
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

def test_Vertex__C(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "_C")
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

def test_Vertex_N_(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "N_")
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

def test_Vertex__N(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "_N")
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

def test_Vertex_C_(bbdb):
   bbs = bbdb.query("all")
   v = Vertex(bbs, "N_")
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
