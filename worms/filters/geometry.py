import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm

from worms.util import jit, InProcessExecutor
from worms.search.result import ResultTable


def check_geometry(ssdag, crit, rslt, **kw):
    heights = np.zeros_like(rslt.err)
    verts = ssdag.verts
    for i, idx in enumerate(rslt.idx):
        coms = [
            rslt.pos[i, k] @ ssdag.bbs[k][verts[k].ibblock[rslt.idx[i, k]]].com
            for k in range(len(ssdag.verts))
        ]
        comz = [c[2] for c in coms]
        heights[i] = np.max(comz) - np.min(comz)
    order = np.argsort(heights)
    r = ResultTable(rslt)
    r.add('height', heights)
    # r.reorder(order)
    return r
