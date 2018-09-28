import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm

from worms.util import jit, InProcessExecutor
from worms.search.result import ResultTable


def check_geometry(ssdag, crit, rslt, **kw):
    zheight = np.zeros_like(rslt.err)
    zradius = np.zeros_like(rslt.err)
    radius = np.zeros_like(rslt.err)
    duplicate = np.ones_like(rslt.err, dtype=np.int32)
    verts = ssdag.verts
    selection = dict()
    for i, idx in enumerate(rslt.idx):

        # prep data
        bb = [
            ssdag.bbs[k][verts[k].ibblock[rslt.idx[i, k]]]
            for k in range(len(ssdag.verts))
        ]
        coms = np.stack([rslt.pos[i, j] @ x.com for j, x in enumerate(bb)])
        names = tuple([bytes(x.name) for x in bb])

        # measure some geometry
        zradius[i] = np.sqrt(np.max([np.sum(c[:2]**2) for c in coms]))
        radius[i] = np.sqrt(np.max([np.sum((c - coms[0])**2) for c in coms]))
        comz = [c[2] for c in coms]
        zheight[i] = np.max(comz) - np.min(comz)

        # find duplicates
        if not names in selection: selection[names] = list()
        selcoms = selection[names]
        mindist2 = 9e9
        for scoms in selcoms:
            mindist2 = min(mindist2, np.sum((scoms - coms)**2))
        if mindist2 > 1.0:
            selcoms.append(coms)
            duplicate[i] = 0

    r = ResultTable(rslt)
    r.add('zheight', zheight)
    r.add('zradius', zradius)
    r.add('radius', radius)

    print(
        'removing', np.sum(duplicate != 0), 'duplicates of', len(duplicate),
        'results'
    )
    # remove duplicates
    r.update(duplicate == 0)

    return r
