import pytest
import pickle
import numpy as np
from homog import hrot, htrans, axis_angle_of, axis_ang_cen_of
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .. import *
from ..criteria_unbounded import *
from homog.sym import icosahedral_axes as IA
import time
try:
    import pyrosetta
    HAVE_PYROSETTA = True
    try:
        import pyrosetta.distributed
        HAVE_PYROSETTA_DISTRIBUTED = True
    except ImportError:
        HAVE_PYROSETTA_DISTRIBUTED = False
except ImportError:
    HAVE_PYROSETTA = HAVE_PYROSETTA_DISTRIBUTED = False


only_if_pyrosetta = pytest.mark.skipif('not HAVE_PYROSETTA')

@only_if_pyrosetta
def test_sheet_P321(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'),
                                       ('2,:2', 'N'), ('2,-2:', 'C'),
                                       ('3,:1', 'N'), ('3,-2:', 'C')])
    segments = [Segment([trimer], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([dimer], 'N_')]
    w = grow(segments, Sheet_P321(c2=-1, c3=0), thresh=1)
    # print(w.scores)
    # vis.show_with_z_axes(w, 0)
    p = w.pose(0, only_connected=0)
    q = w.sympose(0)
    #p.dump_pdb('p.pdb')
    #q.dump_pdb('q.pdb')

    assert util.no_overlapping_residues(p)
    p.dump_pdb("test.pdb")
    assert 0
    # print(len(p))

