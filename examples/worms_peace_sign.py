from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import multiprocessing
import sys
from worms import *
import pyrosetta

# import cProfile
# import pstats

if 1:
    # def main():
    pyrosetta.init("-corrections:beta_nov16 -mute all")
    helix = Spliceable(data.poselib.c1, [(":1", "N"), ("-7:", "C")])
    dimer = Spliceable(
        data.poselib.c2,
        sites=[("1,:1", "N"), ("1,-1:", "C"), ("2,:1", "N"), ("2,-1:", "C")],
    )
    dimerCN = Spliceable(data.poselib.c2, sites=[("1,:1", "N"), ("2,-1:", "C")])
    hub = Spliceable(data.poselib.c3_splay, sites=[("1,:1", "N"), ("1,-1:", "C")])
    trimer = Spliceable(
        data.poselib.c3_splay,
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
        Segment([hub], "_C"),  # origin_seg
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([trimer], "NN"),  # from_seg
        Segment([helix], "CN"),
        Segment([helix], "CN"),
        Segment([helix], "CN"),
        Segment([helix], "CN"),
        Segment([dimer], "CC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([helix], "NC"),
        Segment([trimer], "N_"),
    ]  # to_seg
    from_seg = util.first_duplicate(segments)
    w = grow(
        segments,
        Cyclic(3, from_seg=from_seg, origin_seg=0),
        thresh=2,
        # executor=ThreadPoolExecutor,
        executor=ProcessPoolExecutor,
        max_workers=multiprocessing.cpu_count(),
        memsize=1e6,
        verbosity=0,
        max_samples=1e24,
    )

    # w = Worms(segments, )

    # print(repr(w.scores[:2]))
    # print(repr(w.indices[:2]))
    # print(repr(w.positions[:2]))

    # sys.exit()
    # p = pstats.Stats('grow.stats')
    # p.strip_dirs().sort_stats('time').print_stats(10)
    if w is None:
        print("no results!")
    else:
        print("len(w)", len(w))
        print(w.indices)
        nfail = 0
        for i in range(len(w)):
            try:
                p, s = w.sympose(i, score=True, fullatom=True)
            except:
                import traceback

                traceback.print_exc(file=sys.stdout)
                nfail += 1
                continue
            print(i, w.scores[i], s)
            if s > 100:
                continue
            p.dump_pdb("peace_%04i.pdb" % i)
            w.pose(i, join=0).dump_pdb("peace_%04i_asym.pdb" % i)
            w.pose(i, join=0, cyclic_permute=0).dump_pdb("peace_%04i_asym_nocp.pdb" % i)
            sys.stdout.flush()
        # vis.showme(w.sympose(0))
        # for i in range(0, len(w), multiprocessing.cpu_count()):
        # for p, s in w.sympose(range(i, min(len(w), i + 8)), score=True):
        # print(i, w.scores[i], len(p), s)
        # p.dump_pdb('peace_%04i.pdb' % i)

    print("nfail", nfail, "of", len(w))

# if __name__ == '__main__':
# main()
