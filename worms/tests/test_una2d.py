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
def test_sheet_P6(c2pose, c6pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-7:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    hexamer = Spliceable(c6pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = [Segment([hexamer], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([dimer], 'N_')]
    w = grow(segments, Sheet_P6(c2=-1, c6=0), thresh=1)
    assert len(w) > 0
    p = w.pose(0, only_connected=0)
    assert util.no_overlapping_residues(p)


@only_if_pyrosetta
def test_sheet_P4212(c2pose, c4pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C')])
    tetramer = Spliceable(c4pose, sites=[('1,:1', 'N'), ('1,-2:', 'C')])
    segments = [Segment([tetramer], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([dimer], 'N_')]
    w = grow(segments, Sheet_P4212(c2=-1, c4=0), thresh=1)
    assert len(w) > 0
    # print(w.scores)
    # vis.show_with_z_axes(w, 0)
    p = w.pose(0, only_connected=0)
    q = w.sympose(0, )
    # p.dump_pdb('p.pdb')
    q.dump_pdb('P4212_symm.pdb')
    # p.dump_pdb('P4212_asymm.pdb')

    # basic check on pose to make sure residues are not on top of each other
    assert util.no_overlapping_residues(p)


@only_if_pyrosetta
def test_sheet_P321(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = [Segment([trimer], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([dimer], 'N_')]
    w = grow(segments, Sheet_P321(c2=-1, c3=0), thresh=1)
    assert len(w) > 0
    # print(w.scores)
    # vis.show_with_z_axes(w, 0)
    p = w.pose(0, only_connected=0)
    # q = w.sympose(0, )
    # p.dump_pdb('p.pdb')
    # q.dump_pdb('P321_symm.pdb')
    # p.dump_pdb('P321_asymm.pdb')

    assert util.no_overlapping_residues(p)

    # print(len(p))
