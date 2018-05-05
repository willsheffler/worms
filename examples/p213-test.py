from worms import *
from worms.data import poselib
from worms.vis import showme
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
from worms.criteria_unbounded import *
import sys
import pyrosetta


def test_crystal_P213():
    pyrosetta.init('-corrections:beta_nov16 -mute all -preserve_crystinfo')
    helix = Spliceable(poselib.curved_helix, [(':4', 'N'), ('-4:', 'C')])
    trimer = Spliceable(poselib.c3, sites=[('1,:1', 'N'), ('1,-2:', 'C')])
    trimer2 = Spliceable(poselib.c3_splay, sites=[('1,:1', 'N'), ('1,-2:', 'C')])
    segments = [Segment([trimer], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([trimer2], 'N_')]
    w = grow(segments, Crystal_P213_C3_C3(c3a=0, c3b=-1), thresh=1)
    print(len(w))

    # print(w.scores)
    # vis.show_with_z_axes(w, 0)
    for i in range(1):
        p = w.pose(i, only_connected=0)
        print(p.pdb_info().crystinfo().spacegroup())
        print(p.pdb_info().crystinfo().A())
        print(p.pdb_info().crystinfo().B())
        print(p.pdb_info().crystinfo().C())
        q = w.sympose(i, fullatom=True )
    #p.dump_pdb('p.pdb')
        q.dump_pdb('P213_symm_%i.pdb'%i)
        p.dump_pdb('P213_asymm_%i.pdb'%i)

    assert util.no_overlapping_residues(p) ## basic check on pose to make sure residues are not on top of each other
    assert 0

test_crystal_P213()