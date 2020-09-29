import pytest
from worms import *
from .. import only_if_pyrosetta

@only_if_pyrosetta
def test_NullCriteria(c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   segments = [Segment([helix], "_C"), Segment([helix], "N_")]
   results = grow(segments, NullCriteria())
   assert len(results) == 16
   # vis.showme(results.pose(0))
