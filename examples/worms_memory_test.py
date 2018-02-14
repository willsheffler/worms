from worms import *
from worms.data import poselib
from worms.vis import showme
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import perf_counter
import sys
try:
    import pyrosetta
    HAVE_PYROSETTA = True
except ImportError:
    print('no pyrosetta!')
    HAVE_PYROSETTA = False


def main(nseg, workers=0):
    pyrosetta.init('-corrections:beta_nov16 -mute all')
    helix = Spliceable(poselib.curved_helix, sites=[(1, 'N'), ('-4:', 'C')])
    dimer = Spliceable(poselib.c2, sites=[('1,:1', 'N'), ('2,-1:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([dimer], entry='N', exit='C')] +
                [Segment([helix], entry='N', exit='C')] * (nseg - 2) +
                [Segment([helix], entry='N')])
    t = perf_counter()

    worms = grow(segments,
                 Cyclic('C2', lever=20),
                 thresh=5.01, max_workers=workers,
                 executor=ProcessPoolExecutor,
                 # executor=dask.distributed.Client,
                 max_results=10000, max_samples=10000000)
    print('number of results:', len(worms))
    t = perf_counter() - t
    count = np.prod([len(s) for s in segments])
    s = worms.scores

    try: ptile = np.percentile(s, range(0, 100, 20))
    except: ptile = []
    print('quantile', ptile)
    print('best10  ', s[:10])
    print('nseg %2i' % nseg,
          'best %7.3f' % (s[0] if len(s) else 999),
          'tgrow %7.2f' % t,
          'rate %7.3fM/s' % (count / t / 1000000),
          'npass %8i' % len(s))
    print('going through poses:')
    sys.stdout.flush()

    for i in range(min(10, len(worms))):
        fname = 'worm_nseg%i_%i.pdb' % (nseg, i)
        pose, score0 = worms.sympose(i, fullatom=True, score=True)
        print('worm', i, worms.scores[i], score0, worms.indices[i])
        sys.stdout.flush()
        if score0 < 50:
            print('    dumping to', fname)
            pose.dump_pdb(fname)

if __name__ == '__main__':
    if HAVE_PYROSETTA:
        main(14, workers=7)
