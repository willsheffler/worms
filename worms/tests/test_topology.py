import pytest
from worms.topology import Topology

def test_topology():
   with pytest.raises(AssertionError):
      has_cycle = Topology([0, 1, 1, 0])

   t = Topology([0, 1, 1, 2, 2, 3])
   paths = t.paths()
   assert len(paths) is 1
   assert len(paths[0]) is 4
   assert t.common_prefix() == 4

   t = Topology([0, 1, 1, 2, 1, 3])
   paths = t.paths()
   assert len(paths) is 2
   assert len(paths[0]) is 3
   assert len(paths[1]) is 3
   print(paths)
   assert t.common_prefix() == 2

   t = Topology([0, 1, 1, 2, 1, 3, 3, 4])
   paths = t.paths()
   assert len(paths) is 2
   assert len(paths[0]) is 3
   assert len(paths[1]) is 4
   assert t.common_prefix() == 2

   t = Topology([0, 1, 1, 2, 2, 3, 3, 4, 3, 5, 3, 6])
   paths = t.paths()
   assert len(paths) is 3
   assert len(paths[0]) is 5
   assert len(paths[1]) is 5
   assert t.common_prefix() == 4
