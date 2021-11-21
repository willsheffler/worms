import operator
from worms.criteria import *

from worms.search.old_search import grow
from worms.segments import Spliceable, Segment

from worms.homog import hrot, htrans
from .. import only_if_pyrosetta
from worms.util.rosetta_utils import residue_sym_err
from worms.homog.sym import icosahedral_axes as IA
from worms.util.rosetta_utils import no_overlapping_residues

def test_geom_check():
   SX = Cyclic
   I = np.identity(4).reshape(1, 4, 4)
   rotx1rad = hrot([1, 0, 0], 1).reshape(1, 4, 4)
   transx10 = htrans([10, 0, 0]).reshape(1, 4, 4)
   randaxes = np.random.randn(1, 3)

   assert 0 == SX("c1").score([I, I])
   assert 0.001 > abs(50 - SX("c1").score([I, rotx1rad]))
   assert 1e-5 > abs(SX("c2").score([I, hrot([1, 0, 0], np.pi)]))

   score = Cyclic("c2").score([I, hrot(randaxes, np.pi)])
   assert np.allclose(0, score, atol=1e-5, rtol=1)

   score = Cyclic("c3").score([I, hrot(randaxes, np.pi * 2 / 3)])
   assert np.allclose(0, score, atol=1e-5, rtol=1)

   score = Cyclic("c4").score([I, hrot(randaxes, np.pi / 2)])
   assert np.allclose(0, score, atol=1e-5, rtol=1)

@only_if_pyrosetta
def test_D3(c2pose, c3pose, c1pose):
   # Spliceable: pose + splice site info
   # Segment: list of spliceables
   helix = Spliceable(c1pose, sites=[(":4", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C"), ("2,:2", "N"),
                                     ("2,-1:", "C")])
   trimer = Spliceable(
      c3pose,
      sites=[
         ("1,:1", "N"),
         ("1,-2:", "C"),
         ("2,:2", "N"),
         ("2,-2:", "C"),
         ("3,:1", "N"),
         ("3,-2:", "C"),
      ],
   )
   segments = [
      Segment([trimer], "_C"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([dimer], "N_"),
   ]
   w = grow(segments, D3(c2=-1, c3=0), thresh=1)

   # vis.show_with_z_axes(w, 0)
   # vis.showme(w.pose(0))
   # assert 0
   # print(w.scores[:5])
   # print([len(s) for s in segments])
   # print(w.indices[:5])
   # for i in range(5):
   # vis.showme(w.pose(i, join=False))
   # vis.showme(w.sympose(i))

   # assert 0

   p = w.pose(0, only_connected=0)
   assert no_overlapping_residues(p)
   # print(len(p))

   # print('foo')
   # assert 0

   assert 1 > residue_sym_err(p, 180, 53, 65, 6, axis=[1, 0, 0])
   assert 1 > residue_sym_err(p, 120, 1, 10, 6, axis=[0, 0, 1])
   # assert 0

   segments = [
      Segment([dimer], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([trimer], entry="N"),
   ]
   w = grow(segments, D3(c2=0, c3=-1), thresh=1)
   # print(w.scores)
   # show_with_z_axes(w, 0)
   p = w.pose(4, only_connected=0)
   assert no_overlapping_residues(p)
   # vis.showme(p)
   assert 1 > residue_sym_err(p, 180, 1, 13, 6, axis=[1, 0, 0])
   assert 1 > residue_sym_err(p, 120, 56, 65, 6, axis=[0, 0, 1])

@only_if_pyrosetta
def test_tet(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = ([Segment([dimer], exit="C")] + [Segment([helix], entry="N", exit="C")] * 5 +
               [Segment([trimer], entry="N")])
   w = grow(segments, Tetrahedral(c3=-1, c2=0), thresh=2)
   assert len(w)
   p = w.pose(3, only_connected=0)
   assert no_overlapping_residues(p)
   assert 2.5 > residue_sym_err(p, 120, 86, 95, 6, axis=[1, 1, 1])
   assert 2.5 > residue_sym_err(p, 180, 2, 14, 6, axis=[1, 0, 0])

@only_if_pyrosetta
def test_tet33(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = ([Segment([trimer], exit="C")] + [Segment([helix], entry="N", exit="C")] * 5 +
               [Segment([trimer], entry="N")])
   w = grow(segments, Tetrahedral(c3=-1, c3b=0), thresh=2)
   assert len(w) == 3
   p = w.pose(0, only_connected=0)
   assert no_overlapping_residues(p)
   assert 2.5 > residue_sym_err(p, 120, 2, 20, 6, axis=[1, 1, -1])
   assert 2.5 > residue_sym_err(p, 120, 87, 96, 6, axis=[1, 1, 1])

@only_if_pyrosetta
def test_oct(c2pose, c3pose, c4pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   tetramer = Spliceable(c4pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = ([Segment([dimer], exit="C")] + [Segment([helix], entry="N", exit="C")] * 5 +
               [Segment([trimer], entry="N")])
   w = grow(segments, Octahedral(c3=-1, c2=0), thresh=1)
   assert len(w) == 1
   p = w.pose(0, only_connected=0)
   assert no_overlapping_residues(p)
   assert 1 > residue_sym_err(p, 120, 85, 94, 6, axis=[1, 1, 1])
   assert 1 > residue_sym_err(p, 180, 1, 13, 6, axis=[1, 1, 0])

   segments = ([Segment([tetramer], exit="C")] + [Segment([helix], entry="N", exit="C")] * 5 +
               [Segment([dimer], entry="N")])
   w = grow(segments, Octahedral(c2=-1, c4=0), thresh=1)
   assert len(w) == 5
   assert np.allclose(
      w.indices,
      np.array([
         [0, 1, 1, 2, 0, 2, 1],
         [1, 0, 2, 3, 1, 0, 1],
         [1, 0, 0, 0, 3, 2, 0],
         [0, 2, 0, 0, 1, 2, 0],
         [1, 1, 2, 1, 1, 2, 0],
      ]),
   )
   p = w.pose(0, only_connected=0)
   assert p.sequence() == ("AIAAALAAIAAIAAALAAIAAIAAALAAIAAIAAALAAAAAAAAAAGA" +
                           "AAAAAAAAGAAAAAAAAAGAAAAAAAAAAGAAAAAAAAGAATAFLA" +
                           "AIPAINYTAFLAAIPAIN")
   assert no_overlapping_residues(p)
   # from socket import gethostname
   # p.dump_pdb(gethostname() + '.pdb')
   # assert np.allclose(p.residue(1).xyz('CA')[0], 33.0786722948)
   assert 1 > residue_sym_err(p, 90, 1, 31, 6, axis=[1, 0, 0], verbose=0)
   assert 1 > residue_sym_err(p, 180, 92, 104, 6, axis=[1, 1, 0], verbose=0)
   # assert 0

@only_if_pyrosetta
def test_icos(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = ([Segment([dimer], exit="C")] + [Segment([helix], entry="N", exit="C")] * 5 +
               [Segment([trimer], entry="N")])
   w = grow(segments, Icosahedral(c3=-1, c2=0), thresh=2)
   assert len(w) == 3
   p = w.pose(2, only_connected=0)
   assert no_overlapping_residues(p)
   # vis.showme(p)
   assert 2 > residue_sym_err(p, 120, 90, 99, 6, axis=IA[3])
   assert 2 > residue_sym_err(p, 180, 2, 14, 6, axis=IA[2])
