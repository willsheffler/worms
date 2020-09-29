from functools import partial
import pytest
from worms.segments import Spliceable, Segment
from worms.tests import only_if_pyrosetta
from worms.pose_contortions import reorder_spliced_as_N_to_C

def test_reorder_spliced_as_N_to_C():
   Q = reorder_spliced_as_N_to_C

   with pytest.raises(ValueError):
      Q([[1], [1], [1]], "NC")
   with pytest.raises(ValueError):
      Q([[1], [1], [1]], "CN")
   with pytest.raises(ValueError):
      Q([[1, 1], [1], [1, 1]], "CN")
   with pytest.raises(ValueError):
      Q([], "CN")
   with pytest.raises(ValueError):
      Q([], "")
   with pytest.raises(ValueError):
      Q([[]], "")

   assert Q([[1]], "") == [[1]]
   assert Q([[1, 2]], "") == [[1], [2]]
   assert Q([[1], [2]], "N") == [[1, 2]]
   assert Q([[1, 2], [3]], "N") == [[1], [2, 3]]
   assert Q([[1, 2], [3, 4]], "N") == [[1], [2, 3], [4]]
   assert Q([[1, 2, 3], [4, 5]], "N") == [[1], [2], [3, 4], [5]]
   assert Q([[1], [2]], "C") == [[2, 1]]
   assert Q([[1, 2], [3]], "C") == [[1], [3, 2]]
   assert Q([[1, 2], [3, 4]], "C") == [[1], [3, 2], [4]]
   assert Q([[1, 2, 3], [4, 5]], "C") == [[1], [2], [4, 3], [5]]

   assert Q([[1], [2], [3]], "NN") == [[1, 2, 3]]
   assert Q([[1], [2], [3, 4]], "NN") == [[1, 2, 3], [4]]
   assert Q([[1], [2, 3], [4, 5]], "NN") == [[1, 2], [3, 4], [5]]
   assert Q([[1, 2], [3, 4], [5, 6]], "NN") == [[1], [2, 3], [4, 5], [6]]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "NN") == [
      [1],
      [2],
      [3, 4],
      [5],
      [6, 7],
      [8],
      [9],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "CN") == [
      [1],
      [2],
      [4, 3],
      [5],
      [6, 7],
      [8],
      [9],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "CC") == [
      [1],
      [2],
      [4, 3],
      [5],
      [7, 6],
      [8],
      [9],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "NC") == [
      [1],
      [2],
      [3, 4],
      [5],
      [7, 6],
      [8],
      [9],
   ]

   for n in range(10):
      x = [[i] for i in range(n + 1)]
      y = list(range(n + 1))
      assert Q(x, "N" * n) == [y]
      assert Q(x, "C" * n) == [y[::-1]]
      assert Q([[13, 14]] + x, "N" + "N" * n) == [[13], [14] + y]
      assert Q([[13, 14]] + x, "C" + "C" * n) == [[13], y[::-1] + [14]]
      assert Q([[10, 11, 12]] + x + [[13, 14, 15]], "N" + "N" * n + "N") == [
         [10],
         [11],
         [12] + y + [13],
         [14],
         [15],
      ]
      assert Q([[10, 11, 12]] + x + [[13, 14, 15]], "C" + "C" * n + "C") == [
         [10],
         [11],
         [13] + y[::-1] + [12],
         [14],
         [15],
      ]

   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], "NNN") == [
      [1],
      [2],
      [3, 4],
      [5],
      [6, 7],
      [8],
      [9, 0],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], "CNN") == [
      [1],
      [2],
      [4, 3],
      [5],
      [6, 7],
      [8],
      [9, 0],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], "NCN") == [
      [1],
      [2],
      [3, 4],
      [5],
      [7, 6],
      [8],
      [9, 0],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], "NNC") == [
      [1],
      [2],
      [3, 4],
      [5],
      [6, 7],
      [8],
      [0, 9],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], "NCC") == [
      [1],
      [2],
      [3, 4],
      [5],
      [7, 6],
      [8],
      [0, 9],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [11], [7, 8, 9], [0, 1, 2]], "NCCC") == [
      [1],
      [2],
      [3, 4],
      [5],
      [7, 11, 6],
      [8],
      [0, 9],
      [1],
      [2],
   ]
   assert Q([[1, 2, 3], [4, 5, 6], [11], [12], [7, 8, 9], [0, 1, 2]], "NCCCN") == [
      [1],
      [2],
      [3, 4],
      [5],
      [7, 12, 11, 6],
      [8],
      [9, 0],
      [1],
      [2],
   ]
   assert Q([[1, 2, 5, 5, 3], [4, 5, 6], [11], [12], [7, 8, 9], [0, 1, 2]],
            "NCCCN") == [[1], [2], [5], [5], [3, 4], [5], [7, 12, 11, 6], [8], [9, 0], [1], [2]]

@only_if_pyrosetta
def test_make_pose_chains_dimer(c2pose):

   dimer = Spliceable(
      c2pose,
      sites=[("1,2:2", "N"), ("2,3:3", "N"), ("1,-4:-4", "C"), ("2,-5:-5", "C")],
   )

   seq = dimer.body.sequence()[:12]

   seg = Segment([dimer], "N", None)
   enex, rest = seg.make_pose_chains(0, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[1:], seq]
   assert [x[0].sequence() for x in rest] == []
   assert enex[-1][0] is dimer.chains[2]
   enex, rest = seg.make_pose_chains(1, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[2:], seq]
   assert [x[0].sequence() for x in rest] == []
   assert enex[-1][0] is dimer.chains[1]

   seg = Segment([dimer], "C", None)
   enex, rest = seg.make_pose_chains(0, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[:-3], seq]
   assert [x[0].sequence() for x in rest] == []
   assert enex[-1][0] is dimer.chains[2]
   enex, rest = seg.make_pose_chains(1, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[:-4], seq]
   assert [x[0].sequence() for x in rest] == []
   assert enex[-1][0] is dimer.chains[1]

   seg = Segment([dimer], None, "N")
   enex, rest = seg.make_pose_chains(0, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq, seq[1:]]
   assert [x[0].sequence() for x in rest] == []
   assert enex[0][0] is dimer.chains[2]
   enex, rest = seg.make_pose_chains(1, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq, seq[2:]]
   assert [x[0].sequence() for x in rest] == []
   assert enex[0][0] is dimer.chains[1]

   seg = Segment([dimer], "N", "N")
   enex, rest = seg.make_pose_chains(0, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[1:], seq[2:]]
   assert [x[0].sequence() for x in rest] == []
   enex, rest = seg.make_pose_chains(1, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[2:], seq[1:]]
   assert [x[0].sequence() for x in rest] == []
   with pytest.raises(IndexError):
      enex, rest = seg.make_pose_chains(2, pad=(0, 1))

   seg = Segment([dimer], "N", "C")
   enex, rest = seg.make_pose_chains(0, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[1:-3]]
   assert [x[0].sequence() for x in rest] == [seq]
   assert rest[0][0] is dimer.chains[2]
   enex, rest = seg.make_pose_chains(1, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[1:], seq[:-4]]
   assert [x[0].sequence() for x in rest] == []
   enex, rest = seg.make_pose_chains(2, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[2:], seq[:-3]]
   assert [x[0].sequence() for x in rest] == []
   enex, rest = seg.make_pose_chains(3, pad=(0, 1))
   assert [x[0].sequence() for x in enex] == [seq[2:-4]]
   assert [x[0].sequence() for x in rest] == [seq]
   assert rest[0][0] is dimer.chains[1]
   with pytest.raises(IndexError):
      enex, rest = seg.make_pose_chains(4, pad=(0, 1))
