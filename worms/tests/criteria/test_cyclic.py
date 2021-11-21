from worms.criteria import Cyclic
from worms.search.old_search import grow
from worms.segments import Spliceable, Segment
from .. import only_if_pyrosetta

@only_if_pyrosetta
def test_extra_chain_handling_cyclic(c1pose, c2pose, c3hetpose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:3", "N"), ("1,-3:", "C")])
   trimer = Spliceable(c3hetpose, sites=[("1,:3", "N"), ("2,-3:", "C")])

   segments = [Segment([helix], "_C"), Segment([dimer], "NC"), Segment([helix], "N_")]
   w = grow(segments, Cyclic(9), thresh=3)
   assert len(w) == 1
   assert tuple(w.indices[0]) == (2, 7, 0)
   p, prov = w.pose(0, provenance=1, only_connected=0)
   assert len(prov) == 3
   assert prov[0] == (1, 11, c1pose, 1, 11)
   assert prov[1] == (12, 19, c2pose, 3, 10)
   assert prov[2] == (21, 32, c2pose, 13, 24)
   p, prov = w.pose(0, provenance=1, only_connected=1)
   assert len(prov) == 2
   assert prov[0] == (1, 11, c1pose, 1, 11)
   assert prov[1] == (12, 19, c2pose, 3, 10)
   p, prov = w.pose(0, provenance=1, only_connected="auto")
   assert len(prov) == 3
   assert prov[0] == (1, 11, c1pose, 1, 11)
   assert prov[1] == (12, 19, c2pose, 3, 10)
   assert prov[2] == (21, 32, c2pose, 13, 24)

   segments = [Segment([helix], "_C"), Segment([trimer], "NC"), Segment([helix], "N_")]
   w = grow(segments, Cyclic(6), thresh=3)
   # vis.showme(w.pose(0))
   assert len(w) == 1
   assert tuple(w.indices[0]) == (3, 7, 0)
   p, prov = w.pose(0, provenance=1, only_connected=0)
   assert len(prov) == 4
   assert prov[0] == (1, 7, c3hetpose, 10, 16)
   assert prov[1] == (8, 19, c1pose, 1, 12)
   assert prov[2] == (20, 26, c3hetpose, 3, 9)
   assert prov[3] == (27, 35, c3hetpose, 19, 27)
   p, prov = w.pose(0, provenance=1, only_connected=1)
   assert len(prov) == 3
   assert prov[0] == (1, 7, c3hetpose, 10, 16)
   assert prov[1] == (8, 19, c1pose, 1, 12)
   assert prov[2] == (20, 26, c3hetpose, 3, 9)
   # assert prov[3] == (27, 35, c3hetpose, 19, 27)
   p, prov = w.pose(0, provenance=1, only_connected="auto")
   assert len(prov) == 4
   assert prov[0] == (1, 7, c3hetpose, 10, 16)
   assert prov[1] == (8, 19, c1pose, 1, 12)
   assert prov[2] == (20, 26, c3hetpose, 3, 9)
   assert prov[3] == (27, 35, c3hetpose, 19, 27)
