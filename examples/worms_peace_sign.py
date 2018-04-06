
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import multiprocessing
import sys
from worms import *
import pyrosetta
# import cProfile
# import pstats


def first_duplicate(segs):
    for i in range(len(segs) - 1, 0, -1):
        for j in range(i):
            if segs[i].spliceables == segs[j].spliceables:
                return j
    return None


if 1:
    # def main():
    pyrosetta.init('-corrections:beta_nov16 -mute all')
    helix = Spliceable(data.poselib.c1, [
        (':1', 'N'), ('-7:', 'C')])
    dimer = Spliceable(data.poselib.c2, sites=[
        ('1,:1', 'N'), ('1,-1:', 'C'),
        ('2,:1', 'N'), ('2,-1:', 'C'), ])
    hub = Spliceable(data.poselib.c3_splay, sites=[
        ('1,:1', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(data.poselib.c3_splay, sites=[
        ('1,:1', 'N'), ('1,-1:', 'C'),
        ('2,:1', 'N'), ('2,-1:', 'C'),
        ('3,:1', 'N'), ('3,-1:', 'C'), ])
    segments = [Segment([hub], '_C'),  # origin_seg
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
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([helix], 'CN'),
                Segment([trimer], 'C_'), ]  # to_seg
    w = grow(segments,
             Cyclic(3, from_seg=first_duplicate(segments), origin_seg=0),
             thresh=1,
             # executor=ThreadPoolExecutor,
             executor=ProcessPoolExecutor,
             max_workers=multiprocessing.cpu_count(),
             memsize=1e6, verbosity=0, max_samples=1e24)

    # sys.exit()
    # p = pstats.Stats('grow.stats')
    # p.strip_dirs().sort_stats('time').print_stats(10)
    if w is None:
        print('no results!')
    else:
        print(len(w))
        for i in range(len(w)):
            if i % 2 is 1: continue
            p, s = w.sympose(i, score=True, fullatom=True)
            print(i, w.scores[i], s)
            p.dump_pdb('peace_%04i.pdb' % i)
            sys.stdout.flush()
        # vis.showme(w.sympose(0))
        # for i in range(0, len(w), multiprocessing.cpu_count()):
            # for p, s in w.sympose(range(i, min(len(w), i + 8)), score=True):
            # print(i, w.scores[i], len(p), s)
            # p.dump_pdb('peace_%04i.pdb' % i)

# if __name__ == '__main__':
    # main()
