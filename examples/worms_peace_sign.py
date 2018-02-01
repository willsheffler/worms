from rif.worm import *
from rif.data import poselib
from rif.vis import showme
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import sys


def main():
    helix = Spliceable(poselib.c1, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(poselib.c2, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                          ('2,:1', 'N'), ('2,-1:', 'C'), ])
    trimer = Spliceable(poselib.c3_splay, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                                 ('2,:1', 'N'), ('2,-1:', 'C'),
                                                 ('3,:1', 'N'), ('3,-1:', 'C'), ])
    segments = [Segment([trimer], '_C'),  # origin_seg
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                # Segment([helix], 'NC'),
                # Segment([helix], 'NC'),
                Segment([trimer], 'NN'),  # from_seg
                # Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                # Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([trimer], 'N_'), ]  # to_seg
    w = grow(segments, Cyclic(3, from_seg=4, origin_seg=0), thresh=10,
             executor=ProcessPoolExecutor, max_workers=8, memsize=1e6)
    showme(w.pose(0))
    # for i, err, pose, score0 in w:
    # print(i, err, len(pose), score0)

if __name__ == '__main__':
    main()
