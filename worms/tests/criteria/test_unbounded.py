import pytest, pickle, time
import numpy as np
from worms.homog import hrot, htrans, axis_angle_of, axis_ang_cen_of
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from worms import *
from worms.criteria.unbounded import *
from worms.homog.sym import icosahedral_axes as IA
from worms.segments import Spliceable, Segment
from worms.search.old_search import grow
from .. import only_if_pyrosetta
from worms.util.rosetta_utils import no_overlapping_residues

@only_if_pyrosetta
def test_sheet_P6(c2pose, c6pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-7:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   hexamer = Spliceable(c6pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = [
      Segment([hexamer], "_C"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([dimer], "N_"),
   ]
   w = grow(segments, Sheet_P6(c2=-1, c6=0), thresh=1)
   assert len(w) > 0
   p = w.pose(0, only_connected=0)
   # q = w.sympose(0, )
   # q.dump_pdb('P6_symm.pdb')
   # p.dump_pdb('P6_asymm.pdb')
   assert no_overlapping_residues(p)

@only_if_pyrosetta
def test_sheet_P4212(c2pose, c4pose, c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   tetramer = Spliceable(c4pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = [
      Segment([tetramer], "_C"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([dimer], "N_"),
   ]
   w = grow(segments, Sheet_P4212(c2=-1, c4=0), thresh=1)
   assert len(w) > 0
   # print(w.scores)
   # viz.show_with_z_axes(w, 0)
   p = w.pose(0, only_connected=0)
   # q = w.sympose(0, )
   # q.dump_pdb('P4212_symm.pdb')
   # p.dump_pdb('P4212_asymm.pdb')

   # basic check on pose to make sure residues are not on top of each other
   assert no_overlapping_residues(p)

# @pytest.mark.skip
@only_if_pyrosetta
def test_sheet_P321(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = [
      Segment([trimer], "_C"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([dimer], "N_"),
   ]
   w = grow(segments, Sheet_P321(c2=-1, c3=0), thresh=1)
   assert len(w) > 0
   # print(w.scores)
   # viz.show_with_z_axes(w, 0)
   p = w.pose(0, only_connected=0)
   # q = w.sympose(0, )
   # q.dump_pdb('P321_symm.pdb')
   # p.dump_pdb('P321_asymm.pdb')

   assert no_overlapping_residues(p)

   # print(len(p))

@only_if_pyrosetta
def test_crystal_P213(c3pose, c3_splay_pose, c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   trimer2 = Spliceable(c3_splay_pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   segments = [
      Segment([trimer], "_C"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([trimer2], "N_"),
   ]
   w = grow(segments, Crystal_P213_C3_C3(c3a=0, c3b=-1), thresh=1)
   print(len(w))

   # print(w.scores)
   # viz.show_with_z_axes(w, 0)
   for i in range(1):
      p = w.pose(i, only_connected=0)
      # q = w.sympose(i, fullatom=True)
      # p.dump_pdb('p.pdb')
      # q.dump_pdb('P213_symm_%i.pdb' % i)
      # p.dump_pdb('P213_asymm_%i.pdb' % i)
      assert no_overlapping_residues(p)
   # basic check on pose to make sure residues are not on top of each other
