
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import multiprocessing
import sys
from worms import *
import pyrosetta


def main():
    pyrosetta.init('-corrections:beta_nov16 -mute all')
    helix = Spliceable(data.poselib.c1, [
        (':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(data.poselib.c2, sites=[
        ('1,:1', 'N'), ('1,-1:', 'C'),
        ('2,:1', 'N'), ('2,-1:', 'C'), ])
    trimer = Spliceable(data.poselib.c3_splay, sites=[
        ('1,:1', 'N'), ('1,-1:', 'C'),
        ('2,:1', 'N'), ('2,-1:', 'C'),
        ('3,:1', 'N'), ('3,-1:', 'C'), ])
    segments = [Segment([trimer], '_C'),  # origin_seg
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([trimer], 'NN'),  # from_seg
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([trimer], 'N_'), ]  # to_seg
    w = grow(segments, Cyclic(3, from_seg=8, origin_seg=0),
             thresh=5,
             # executor=ThreadPoolExecutor,
             executor=ProcessPoolExecutor,
             max_workers=multiprocessing.cpu_count(),
             memsize=1e6, verbosity=1, max_samples=1e12)
    if w is None:
        print('no results!')
    else:
        print(len(w))
        # vis.showme(w.pose(0))
        for i in range(0, len(w), multiprocessing.cpu_count()):
            for p, s in w.sympose(range(i, i + 8), score=True):
                print(i, w.scores[i], len(p), s)
                pose.dump_pdb('peace_%04i.pdb' % i)

if __name__ == '__main__':
    main()
