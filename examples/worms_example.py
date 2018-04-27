from worms import *
from worms.data import poselib
from worms.vis import showme
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import sys
import pyrosetta


def main(nseg, nworker=0):
    pyrosetta.init('-corrections:beta_nov16 -mute all')
    helix = Spliceable(poselib.curved_helix, sites=[(1, 'N'), ('-4:', 'C')])
    helix2 = Spliceable(poselib.curved_helix, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([
        Segment([helix], exit='C'),
    ] + [Segment([helix, helix2], entry='N', exit='C')] *
                (nseg - 2) + [Segment([helix], entry='N')])
    t = perf_counter()
    worms = grow(
        segments,
        Cyclic('C2', lever=20),
        thresh=10,
        max_workers=nworker,
        executor=ProcessPoolExecutor)
    t = perf_counter() - t
    count = np.prod([len(s) for s in segments])
    s = worms.scores

    try:
        ptile = np.percentile(s, range(0, 100, 20))
    except:
        ptile = []
    print('quantile', ptile)
    print('best10  ', s[:10])
    print('nseg %2i' % nseg, 'best %7.3f' % (s[0] if len(s) else 999),
          'tgrow %7.2f' % t, 'rate %7.3fM/s' % (count / t / 1000000),
          'npass %8i' % len(s))
    sys.stdout.flush()

    for i in range(min(10, len(worms))):
        print(i)
        worms.pose(i).dump_pdb('worm_nseg%i_%i.pdb' % (nseg, i))


if __name__ == '__main__':
    main(9)
