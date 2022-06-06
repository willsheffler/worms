import pytest
from worms.criteria import NullCriteria
from worms.segments import Spliceable, Segment
from worms.search.old_search import grow
from .. import only_if_pyrosetta

@only_if_pyrosetta
def test_NullCriteria(c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   segments = [Segment([helix], "_C"), Segment([helix], "N_")]
   results = grow(segments, NullCriteria())
   assert len(results) == 16
   # viz.showme(results.pose(0))
