from worms import *
from worms.data import poselib
from worms.vis import showme
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import sys
import pyrosetta


def main():
    pyrosetta.init('-corrections:beta_nov16 -mute all')

    helix0 = Spliceable(
        poselib.curved_helix, [
            ('2:2', 'N'), ('11:11', "C")])
    helix = Spliceable(poselib.curved_helix, [(':4', 'N'), ('-4:', "C")])
    dimer = Spliceable(poselib.c2, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                          ('2,:1', 'N'), ('2,-1:', 'C')])
    c3het = Spliceable(poselib.c3het, sites=[
        ('1,2:2', 'N'), ('2,2:2', 'N'), ('3,2:2', 'N')])
    segments = [Segment([helix0], '_C'),
                Segment([helix0], 'NC'),
                Segment([helix0], 'NC'),
                Segment([c3het], 'NN'),
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([c3het], 'N_'), ]
    w = grow(segments, Cyclic(3, from_seg=3), thresh=1)
    print(w.scores)
    p, sc = w.sympose(0, score=True, fullatom=True)
    print('score is', sc)
    assert sc < 10
    p.dump_pdb('cool_worms_thing.pdb')


if __name__ == '__main__':
    main()
