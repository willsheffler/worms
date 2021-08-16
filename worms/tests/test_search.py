import pytest
import _pickle as pickle
import numpy as np
from worms.homog import hrot, htrans, axis_angle_of, axis_ang_cen_of
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from worms import *
import time
from worms.tests import only_if_pyrosetta, only_if_pyrosetta_distributed
from worms.util import residue_sym_err

@only_if_pyrosetta
def test_grow_cycle(c1pose):
   helix = Spliceable(c1pose, sites=[(1, "N"), ("-4:", "C")])
   segments = ([Segment([helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix], entry="N")])
   worms = grow(segments, Cyclic("C2", lever=20), thresh=20)
   assert 0.1411 < np.min(worms.scores) < 0.1412

@only_if_pyrosetta
def test_grow_cycle_thread_pool(c1pose):
   helix = Spliceable(c1pose, sites=[(1, "N"), ("-4:", "C")])
   segments = ([Segment([helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix], entry="N")])
   worms = grow(segments, Cyclic("C2", lever=20), executor=ThreadPoolExecutor, max_workers=2)
   assert 0.1411 < np.min(worms.scores) < 0.1412
   assert np.sum(worms.scores < 0.1412) == 4

@only_if_pyrosetta
def test_sym_bug(c1pose, c2pose):
   helix = Spliceable(c1pose, sites=[((1, 2, 3), "N"), ((9, 10, 11, 13), "C")])
   dimer = Spliceable(c2pose, sites=[((1, 2, 3), "N", 1), ("1,-1:", "C"), ("2,-1:", "C")])
   segdimer = Segment([dimer], entry="N", exit="C")
   segments = [
      Segment([helix], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N"),
   ]
   wnc = grow(segments, Cyclic(3, lever=200), thresh=1, verbosity=1)
   assert len(wnc) == 3
   print(wnc.scores)
   p = wnc.pose(0, align=1, end=1)
   # vis.showme(p)
   # show_with_axis(wnc, 0)
   # assert 0
   # q = wnc.pose(4)
   # vis.showme(p, name='carterr')
   # vis.showme(q, name='angerr')
   assert residue_sym_err(wnc.pose(0, end=True), 120, 2, 46, 6) < 1.0

@only_if_pyrosetta_distributed
def test_grow_cycle_process_pool(c1pose):
   helix = Spliceable(c1pose, sites=[(1, "N"), ("-4:", "C")])
   segments = ([Segment([helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix], entry="N")])
   worms = grow(segments, Cyclic("C2", lever=20), executor=ProcessPoolExecutor, max_workers=2)
   assert 0.1411 < np.min(worms.scores) < 0.1412
   assert np.sum(worms.scores < 0.1412) == 4

@only_if_pyrosetta
def test_grow_errors(c1pose):
   nsplice = SpliceSite(sele=[1, 2, 3, 4, 5, 6], polarity="N")
   csplice = SpliceSite(sele=[13], polarity="C")
   spliceable1 = Spliceable(body=c1pose, sites=[nsplice, csplice])
   spliceable2 = Spliceable(body=c1pose, sites=[nsplice, csplice])
   spliceables = [spliceable1]
   segments = ([Segment(spliceables, exit="C")] + [Segment(spliceables, "N", "C")] * 3 +
               [Segment(spliceables, entry="N")])
   checkc3 = Cyclic("C2", from_seg=0, to_seg=-1)

   # make sure incorrect begin/end throws error
   with pytest.raises(ValueError):
      grow(segments[:2], criteria=checkc3)
   with pytest.raises(ValueError):
      grow(segments[1:], criteria=checkc3)
   segments_polarity_mismatch = [
      Segment(spliceables, exit="C"),
      Segment(spliceables, entry="C"),
   ]
   with pytest.raises(ValueError):
      grow(segments_polarity_mismatch, criteria=checkc3)

@only_if_pyrosetta
def test_memsize(c1pose):
   helix = Spliceable(c1pose, sites=[((1, 2), "N"), ("-2:", "C")])
   segments = ([Segment([helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix], entry="N")])
   beg = 3
   for i in range(beg, 7):
      w1 = grow(segments, Cyclic("c2"), memsize=10**i, thresh=30)
      assert i == beg or len(w0.scores) == len(w1.scores)
      assert i == beg or np.allclose(w0.scores, w1.scores)
      w0 = w1

@only_if_pyrosetta
def test_pose_alignment_0(c1pose):
   helix = Spliceable(c1pose, sites=[(1, "N"), ("-4:", "C")])
   segments = ([Segment([helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix], entry="N")])
   w = grow(segments, Cyclic("c2"), thresh=1)
   assert len(w)
   print(w.indices)
   for i in range(4):
      assert tuple(w.indices[i]) in (
         (0, 2, 1, 2, 0),
         (2, 1, 2, 0, 0),
         (1, 2, 0, 2, 0),
         (2, 0, 2, 1, 0),
      )
   pose = w.pose(0, align=1, end=1)
   assert util.no_overlapping_residues(pose)
   # vis.showme(pose)
   xyz0 = np.array([pose.residue(1).xyz(2)[i] for i in (0, 1, 2)] + [1])
   # resid 43 happens to be the symmetrically related one for this solution
   xyz1 = np.array([pose.residue(42).xyz(2)[i] for i in (0, 1, 2)] + [1])
   xyz1 = hrot([0, 0, 1], 180) @ xyz1
   assert np.sum((xyz1 - xyz0)**2) < 0.1

@only_if_pyrosetta
def test_last_body_same_as(c1pose):
   helix = Spliceable(c1pose, sites=[(1, "N"), ("-4:", "C")])
   segments = ([Segment([helix, helix], exit="C")] + [Segment([helix], "N", "C")] * 3 +
               [Segment([helix, helix], entry="N")])
   w = grow(segments, Cyclic("c2"), thresh=1)
   for i, s in zip(w.indices, w.scores):
      assert segments[0].bodyid[i[0]] == segments[-1].bodyid[i[-1]]
   assert len(w) == 8
   ref = [
      (1, 2, 0, 2, 0),
      (5, 2, 0, 2, 1),
      (2, 0, 2, 1, 0),
      (6, 0, 2, 1, 1),
      (0, 2, 1, 2, 0),
      (4, 2, 1, 2, 1),
      (2, 1, 2, 0, 0),
      (6, 1, 2, 0, 1),
   ]
   for i in range(8):
      assert tuple(w.indices[i]) in ref

@only_if_pyrosetta
def test_multichain_match_reveres_pol(c1pose, c2pose):
   helix = Spliceable(c1pose, sites=[((1, 2, 3), "N"), ((9, 10, 11, 13), "C")])
   dimer = Spliceable(c2pose, sites=[("1,:1", "N"), ("1,-1:", "C"), ("2,:2", "N"),
                                     ("2,-1:", "C")])
   segments = [
      Segment([helix], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N"),
   ]
   wnc = grow(segments, Cyclic("C3", lever=20), thresh=1)
   assert len(wnc)
   assert wnc.scores[0] < 0.25

   segments = [
      Segment([helix], exit="N"),
      Segment([helix], entry="C", exit="N"),
      Segment([dimer], entry="C", exit="N"),
      Segment([helix], entry="C", exit="N"),
      Segment([helix], entry="C"),
   ]
   wcn = grow(segments, Cyclic("C3", lever=20), thresh=1)
   # assert residue_sym_err(wcn.pose(0), 120, 22, 35, 8) < 0.5
   # N-to-C and C-to-N construction should be same
   assert np.allclose(wnc.scores, wcn.scores, atol=1e-3)

@only_if_pyrosetta
def test_splicepoints(c1pose, c2pose, c3pose):
   helix = Spliceable(c1pose, sites=[((1, 2, 3), "N"), ((9, 10, 11, 13), "C")])
   dimer = Spliceable(c2pose, sites=[("1,:1", "N"), ("1,-1:", "C"), ("2,:2", "N"),
                                     ("2,-1:", "C")])
   segments = [
      Segment([helix], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N"),
   ]
   w = grow(segments, Cyclic("C3", lever=20), thresh=1)
   assert len(w) == 17
   assert w.scores[0] < 0.25
   assert w.splicepoints(0) == [11, 19, 27, 37]
   w.pose(0, cyclic_permute=0)
   assert w.splicepoints(0) == [10, 20, 42]

   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
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
      Segment([trimer], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N"),
   ]
   w = grow(segments, D3(c2=-1, c3=0), thresh=1)
   assert len(w) == 90
   assert w.splicepoints(0) == [8, 16, 25, 34]

   actual_chains = list(w.pose(0, join=0).split_by_chain())
   for i, splice in enumerate(w.splices(0)):
      ib1, ic1, ir1, ib2, ic2, ir2, dr = splice
      pose1 = w.segments[i].spliceables[ib1].chains[ic1]
      pose2 = w.segments[i + 1].spliceables[ib2].chains[ic2]
      seq1 = str(util.subpose(pose1, 1, ir1 - 1).sequence())
      seq2 = str(util.subpose(pose2, ir2).sequence())
      # print(i, '1', seq1, str(actual_chains[i].sequence()))
      # print(i, '2', seq2, str(actual_chains[i + 1].sequence()))
      assert seq1.endswith(str(actual_chains[i].sequence()))
      assert seq2.startswith(str(actual_chains[i + 1].sequence()))

@only_if_pyrosetta
def test_cyclic_permute_beg_end(c1pose, c2pose):
   helix = Spliceable(c1pose, sites=[((1, 2, 3), "N"), ((9, 10, 11, 13), "C")])
   dimer = Spliceable(c2pose, sites=[("1,:1", "N"), ("1,-1:", "C"), ("2,:2", "N"),
                                     ("2,-1:", "C")])
   segments = [
      Segment([helix], exit="N"),
      Segment([helix], entry="C", exit="N"),
      Segment([dimer], entry="C", exit="N"),
      Segment([helix], entry="C", exit="N"),
      Segment([helix], entry="C"),
   ]
   w = grow(segments, Cyclic("C3", lever=50), thresh=1)
   # vis.showme(w.pose(0))
   p = w.pose(0, cyclic_permute=1)
   assert p.sequence() == "YTAFLAAIPAINAAAAAAAGAAAAAGAAAAAAAGAAAAAFLAAIPAIN"
   assert p.chain(30) == 1
   assert util.no_overlapping_residues(p)

   segments = [
      Segment([helix], "_C"),
      Segment([helix], "NC"),
      Segment([dimer], "NC"),
      Segment([helix], "NC"),
      Segment([helix], "N_"),
   ]
   w = grow(segments, Cyclic("C3", lever=50), thresh=1)
   p = w.pose(0, cyclic_permute=1)
   assert p.sequence() == "YTAFLAAIPAIAAAAAAAAAAAAAAGAAAAAAAGAAATAFLAAIPAIN"
   assert p.chain(len(p)) == 1
   assert util.no_overlapping_residues(p)
   # print(w.scores)
   # vis.showme(w.pose(0, cyclic_permute=0), name='reg')
   # print('------------------------')
   # vis.showme(w.pose(0, end=1, join=False), name='end')
   # print('------------------------')
   # vis.showme(w.pose(0, cyclic_permute=1), name='cp')
   # print('------------------------')
   # assert 0

@only_if_pyrosetta
def test_cyclic_permute_mid_end(c1pose, c2pose, c3hetpose):
   helix0 = Spliceable(c1pose, [([2], "N"), ([11], "C")])
   helix = Spliceable(c1pose, [([1, 3, 4], "N"), ([12], "C")])
   dimer = Spliceable(c2pose, sites=[("1,-1:", "C"), ("2,-1:", "C")], allowed_pairs=[(0, 1)])
   c3het = Spliceable(
      c3hetpose,
      sites=[("1,2:2", "N"), ("2,2:2", "N"), ("3,2:2", "N")],
      allowed_pairs=[(0, 1), (1, 0)],
   )
   segments = [
      Segment([helix0], "_C"),
      Segment([helix0], "NC"),
      Segment([helix0], "NC"),
      Segment([c3het], "NN"),
      Segment([helix], "CN"),
      Segment([dimer], "CC"),
      Segment([helix], "NC"),
      Segment([helix], "NC"),
      Segment([c3het], "N_"),
   ]
   w = grow(segments, Cyclic(3, from_seg=3), thresh=1)
   p, sc = w.sympose(0, score=True)
   assert sc < 4
   assert len(p) == 312
   assert p.chain(306) == 9
   assert util.no_overlapping_residues(p)
   assert len(w) == 1

@only_if_pyrosetta
def test_multichain_mixed_pol(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ((10, 12, 13), "C")])
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
      Segment([helix], exit="C"),
      Segment([dimer], entry="N", exit="N"),
      Segment([helix], entry="C", exit="N"),
      Segment([trimer], entry="C", exit="C"),
      Segment([helix], entry="N"),
   ]
   w = grow(segments, Cyclic("C3"), thresh=1)
   assert len(w) == 24
   p = w.pose(0, end=True, cyclic_permute=0)
   assert util.no_overlapping_residues(p)
   # vis.show_with_axis(w, 0)
   # vis.showme(p)

   # print(residue_sym_err(p, 120, 2, 62, 7))
   assert 0.3 > residue_sym_err(p, 120, 2, 62, 7)

@only_if_pyrosetta
def test_multichain_db(c2pose, c1pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,-1:", "C"), ("2,-1:", "C")])
   segments = [
      Segment([helix], exit="N"),
      Segment([dimer], entry="C", exit="C"),
      Segment([helix], entry="N"),
   ]
   with pytest.raises(ValueError):
      w = grow(segments, Cyclic("C4"), thresh=20)

@only_if_pyrosetta
def test_score0_sym(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ((-4, -3, -2), "C")])
   dimer = Spliceable(c2pose, sites=[((2, ), "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ((2, ), "C")])
   segments = ([Segment([dimer], "_C")] + [Segment([helix], "NC")] * 4 +
               [Segment([trimer], "N_")])
   w = grow(segments, D3(c3=-1, c2=0), thresh=2)
   assert len(w) == 2
   i, err, pose, score0 = w[0]
   # vis.showme(w.pose(1, fullatom=True))
   # show_with_z_axes(w, 1)
   assert 22.488 < score0 < 22.4881
   assert util.no_overlapping_residues(pose)

   if hasattr(pose, "__getstate__"):
      t = time.time()
      ps1 = w.sympose(range(len(w)), score=1)
      t = time.time() - t
      print(t)

      t = time.time()
      ps2 = w.sympose(range(len(w)), score=1, parallel=True)
      t = time.time() - t
      print(t)
      assert np.allclose([x[1] for x in ps1], [x[1] for x in ps2])

@only_if_pyrosetta
def test_chunk_speed(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-2:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   nseg = 11
   segments = ([Segment([dimer], exit="C")] + [Segment([helix], entry="N", exit="C")] *
               (nseg - 2) + [Segment([trimer], entry="N")])
   # w = grow(segments, Tetrahedral(c3=-1, c2=0), thresh=5)
   t1 = time.time()
   w1 = grow(segments, Octahedral(c3=-1, c2=0), thresh=1, memsize=0)
   t1 = time.time() - t1
   t2 = time.time()
   w2 = grow(segments, Octahedral(c3=-1, c2=0), thresh=1, memsize=1e7)
   t2 = time.time() - t2

   print("chunksize", w1.detail["chunksize"], "time", t1)
   print("chunksize", w2.detail["chunksize"], "time", t2)
   print("speedup:", t1 / t2)

   assert t1 / t2 > 10.0  # conservative, but still sketchy...

@only_if_pyrosetta
def test_splice_compatibility_check(c1pose, c2pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-2:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:2", "N"), ("2,:2", "N")])
   segments = [Segment([helix], "_C"), Segment([dimer], "NN"), Segment([helix], "C_")]
   with pytest.raises(ValueError):
      w = grow(segments, Cyclic(), thresh=1)

@only_if_pyrosetta
def test_invalid_splices_seg_too_small(c1pose):
   helix = Spliceable(c1pose, [("8:8", "N"), ("7:7", "C")])
   with pytest.raises(ValueError):
      segments = [
         Segment([helix], "_C"),
         Segment([helix], "NC"),
         Segment([helix], "N_"),
      ]

   helix = Spliceable(c1pose, [("7:8", "N"), ("7:8", "C")])
   segments = [Segment([helix], "_C"), Segment([helix], "NC"), Segment([helix], "N_")]
   w = grow(segments, Cyclic("C3"), thresh=9e9)
   assert len(w) == 12

   helix = Spliceable(c1pose, [("7:8", "N"), ("7:8", "C")], min_seg_len=2)
   segments = [Segment([helix], "_C"), Segment([helix], "NC"), Segment([helix], "N_")]
   w = grow(segments, Cyclic("C3"), thresh=9e9)
   assert len(w) == 4

@only_if_pyrosetta
def test_invalid_splices_site_overlap_2(c1pose, c2pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-1:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:1", "N"), ("2,:1", "N"), ("1,-1:", "C"),
                                     ("2,-1:", "C")])
   segments = [
      Segment([helix], "_C"),
      Segment([dimer], "NN"),
      Segment([helix], "CN"),
      Segment([dimer], "CC"),
      Segment([helix], "N_"),
   ]
   w = grow(segments, Cyclic(3), thresh=9e9)
   assert len(w) == 4
   for i in range(len(w)):
      assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
              w.segments[1].exitsiteid[w.indices[i, 1]])
      assert (w.segments[3].entrysiteid[w.indices[i, 3]] !=
              w.segments[3].exitsiteid[w.indices[i, 3]])

@only_if_pyrosetta
def test_invalid_splices_site_overlap_3(c1pose, c3pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-1:", "C")])
   trimer = Spliceable(
      c3pose,
      sites=[
         ("1,:1", "N"),
         ("1,-1:", "C"),
         ("2,:1", "N"),
         ("2,-1:", "C"),
         ("3,:1", "N"),
         ("3,-1:", "C"),
      ],
   )
   segments = [
      Segment([helix], "_C"),
      Segment([trimer], "NN"),
      Segment([helix], "CN"),
      Segment([trimer], "CC"),
      Segment([helix], "NC"),
      Segment([trimer], "N_"),
   ]
   w = grow(segments, Cyclic(3, from_seg=1), thresh=9e9)
   assert len(w)
   for i in range(len(w)):
      assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
              w.segments[1].exitsiteid[w.indices[i, 1]])
      assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
              w.segments[5].entrysiteid[w.indices[i, 5]])
      assert (w.segments[1].exitsiteid[w.indices[i, 1]] !=
              w.segments[5].entrysiteid[w.indices[i, 5]])

@only_if_pyrosetta
def test_provenance(c1pose):
   sites = [(":1", "N"), ("-4:", "C")]
   segments = [
      Segment([Spliceable(c1pose.clone(), sites)], "_C"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "NC"),
      Segment([Spliceable(c1pose.clone(), sites)], "N_"),
   ]
   w = grow(segments, Cyclic(6), thresh=2, expert=True)
   assert len(w)
   for i in range(len(w)):
      # pose, score, srcpose, srcres = w.sympose(
      # i, score=True, provenance=True)
      pose, prov = w.pose(i, provenance=True)

      for i, prv in enumerate(prov):
         lb, ub, src_pose, src_lb, src_ub = prv
         assert src_pose is segments[i].spliceables[0].body
         assert src_pose is not c1pose
         srcseq = src_pose.sequence()[src_lb - 1:src_ub]
         seq = pose.sequence()[lb - 1:ub]
         assert srcseq == seq
      assert len(prov) == len(segments) - 1

@only_if_pyrosetta
def test_extra_chain_handling_noncyclic(c1pose, c2pose, c3pose, c3hetpose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
   dimer = Spliceable(c2pose, sites=[("1,:1", "N"), ("1,-1:", "C")])
   trimer = Spliceable(c3pose, sites=[("1,:1", "N"), ("1,-2:", "C")])
   hettri = Spliceable(c3hetpose, sites=[("1,:1", "N"), ("1,-1:", "C")])
   segments = [
      Segment([trimer], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([hettri], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N"),
   ]
   w = grow(segments, D3(c2=-1, c3=0), thresh=1)
   # vis.showme(w.sympose(0, fullatom=1))
   assert len(w) == 4
   assert w.pose(0, only_connected="auto").num_chains() == 3
   assert w.pose(0, only_connected=0).num_chains() == 6
   assert w.pose(0, only_connected=1).num_chains() == 1

   hettri = Spliceable(c3hetpose, sites=[("1,:1", "N"), ("2,-1:", "C")])
   segments = [
      Segment([trimer], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([hettri], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N"),
   ]
   w = grow(segments, D3(c2=-1, c3=0), thresh=1)
   assert len(w) == 1
   assert w.pose(0, only_connected="auto").num_chains() == 3
   assert w.pose(0, only_connected=0).num_chains() == 6
   assert w.pose(0, only_connected=1).num_chains() == 2

@only_if_pyrosetta
def test_max_results(c1pose, c2pose, c3pose):
   helix = Spliceable(c1pose, [(":4", "N"), ("-4:", "C")])
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
      Segment([trimer], exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([helix], entry="N", exit="C"),
      Segment([dimer], entry="N"),
   ]
   wref = grow(segments, D3(c2=-1, c3=0), thresh=1)
   assert len(wref) == 90

   s = wref.scores[:]
   s.sort()
   i = np.argmin(s[1:] - s[:-1])

   wtst = grow(segments, D3(c2=-1, c3=0), thresh=1, max_results=90)
   assert len(wtst) == 90

   assert np.all(wref.indices == wtst.indices)

@only_if_pyrosetta
def test_chunk_speed(c2pose, c3pose, c1pose):
   helix = Spliceable(c1pose, [(":1", "N"), ("-4:", "C")])
   nseg = 39
   segments = ([Segment([helix], exit="C")] + [Segment([helix], entry="N", exit="C")] *
               (nseg - 2) + [Segment([helix], entry="N")])
   with pytest.raises(ValueError):
      grow(segments, Octahedral(c3=-1, c2=0), thresh=1, max_samples=1000000)
