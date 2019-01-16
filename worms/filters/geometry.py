import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm

import homog.sym as sym
from worms.util import jit, InProcessExecutor
from worms.search.result import ResultTable
from worms.data.sphere import get_sphere_samples
from worms.filters.clash import _chain_bounds
from worms.vis.plot import plot3d, scatter


def check_geometry(ssdag, crit, rslt, max_porosity, max_com_redundancy, **kw):
    zheight = np.zeros_like(rslt.err)
    zradius = np.zeros_like(rslt.err)
    radius = np.zeros_like(rslt.err)
    porosity = -np.ones_like(rslt.err)
    remove_me = np.ones_like(rslt.err, dtype=np.int32)
    verts = ssdag.verts

    if max_porosity < 9e8 and crit.symname not in ('T', 'O', 'I'):
        print('check_geometry: max_porosity only supported with sym T, O, I')
    symframes = dict(
        T=sym.tetrahedral_frames,
        O=sym.octahedral_frames,
        I=sym.icosahedral_frames
    )

    selection = dict()
    for i, idx in enumerate(rslt.idx):

        # prep data
        bb = [
            ssdag.bbs[k][verts[k].ibblock[rslt.idx[i, k]]]
            for k in range(len(ssdag.verts))
        ]
        xaln = crit.alignment(rslt.pos[i])
        coms = np.stack([rslt.pos[i, j] @ x.com for j, x in enumerate(bb)])
        coms = xaln @ coms[..., None]
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
        if mindist2 > max_com_redundancy**2:
            selcoms.append(coms)
            remove_me[i] = 0

        if remove_me[i]: continue
        if max_porosity >= 9e8: continue
        if crit.symname not in ('T', 'O', 'I'): continue

        # build ca coords
        ca = list()
        for iseg, bblock in enumerate(bb):
            chains = _chain_bounds(
                dirn=ssdag.verts[iseg].dirn,
                ires=ssdag.verts[iseg].ires[idx[iseg]],
                chains=bblock.chains,
                spliced_only=iseg in (crit.from_seg, crit.to_seg),
                trim=0
            )
            for lb, ub in chains:
                # print('ca', iseg, lb, ub)
                ca.append(rslt.pos[i, iseg] @ bblock.ncac[lb:ub, 1, :, None])
        ca = np.concatenate(ca)
        ca = xaln @ ca

        # porosity check

        frames = symframes[crit.symname]
        symca = frames[:, None] @ ca
        symca = symca[:, :, :3].squeeze().reshape(-1, 3)
        normca = symca / np.linalg.norm(symca, axis=1)[:, None]

        # compute fraction of sph points overlapped by protein
        # sph is ~1degree covering radius grid
        sph = get_sphere_samples(sym=crit.symname)
        d2 = np.sum((sph[:, None] - normca)**2, axis=2)
        md2 = np.min(d2, axis=1)
        sphere_surface = 4 * np.pi * radius[i]**2
        porosity[i] = sum(md2 > 0.002) / len(sph) * sphere_surface
        if porosity[i] > max_porosity:
            remove_me[i] = 1

    r = ResultTable(rslt)
    r.add('zheight', zheight)
    r.add('zradius', zradius)
    r.add('radius', radius)
    r.add('porosity', porosity)

    print(
        f'mbb{kw["merge_bblock"]:04} geometry: removing',
        np.sum(remove_me != 0), ' of', len(remove_me), 'results'
    )
    # remove duplicates
    r.update(remove_me == 0)

    return r
