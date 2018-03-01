import os
import functools as ft
import itertools as it
import operator
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
import multiprocessing
try:
    from pyrosetta import rosetta as ros
except ImportError:
    pass


def cpu_count():
    try: return int(os.environ['SLURM_CPUS_ON_NODE'])
    except: return multiprocessing.cpu_count()


class WormsAccumulator:

    def __init__(self, max_results=1000000, max_tmp_size=1024):
        self.max_tmp_size = max_tmp_size
        self.max_results = max_results
        self.temporary = []

    def accumulate_sort_filter(self):
        if len(self.temporary) is 0: return
        if hasattr(self, 'scores'):
            sc, li, lp = [self.scores], [self.lowidx], [self.lowpos]
        else:
            sc, li, lp = [], [], []
        scores = np.concatenate([x[0] for x in self.temporary] + sc)
        lowidx = np.concatenate([x[1] for x in self.temporary] + li)
        lowpos = np.concatenate([x[2] for x in self.temporary] + lp)
        order = np.argsort(scores)
        self.scores = scores[order[:self.max_results]]
        self.lowidx = lowidx[order[:self.max_results]]
        self.lowpos = lowpos[order[:self.max_results]]
        self.temporary = []

    def accumulate(self, gen):
        for future in gen:
            self.accumulate_future(future)
            yield None

    def accumulate_future(self, future):
        result = future.result()
        if result is not None:
            self.temporary.append(result)
            if len(self.temporary) >= self.max_tmp_size:
                self.accumulate_sort_filter()
        # future._result = None  # doesn't fix memory issue

    def final_result(self):
        # print('batches:', len(self.batches))
        # print('batches len', [len(b) for b in self.batches])
        self.accumulate_sort_filter()
        try:
            return self.scores, self.lowidx, self.lowpos
        except AttributeError:
            return None


def parallel_batch_map(pool, function, accumulator,
                       batch_size, map_func_args, **kw):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    njobs = len(map_func_args[0])
    args = list(zip(*map_func_args))
    for ibatch in range(0, njobs, batch_size):
        beg = ibatch
        end = min(njobs, ibatch + batch_size)
        batch_args = args[beg:end]  # todo, this could be done lazily...
        futures = [pool.submit(function, *a) for a in batch_args]
        if isinstance(pool, (ProcessPoolExecutor, ThreadPoolExecutor)):
            as_completed = cf_as_completed
        else:
            from dask.distributed import as_completed as dd_as_completed
            as_completed = dd_as_completed
        for _ in accumulator.accumulate(as_completed(futures)):
            yield None
        accumulator.accumulate_sort_filter()


def parallel_nobatch_map(pool, function, accumulator,
                         batch_size, map_func_args, **kw):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    njobs = len(map_func_args[0])
    args = list(zip(*map_func_args))
    futures = [pool.submit(function, *a) for a in args]
    if isinstance(pool, (ProcessPoolExecutor, ThreadPoolExecutor)):
        as_completed = cf_as_completed
    else:
        as_completed = dd_as_completed
    for _ in accumulator.accumulate(as_completed(futures)):
        yield None
    accumulator.accumulate_sort_filter()


def tqdm_parallel_map(pool, function, accumulator,
                      map_func_args, batch_size, **kw):
    for _ in tqdm(parallel_batch_map(pool, function, accumulator, batch_size,
                                     map_func_args=map_func_args, **kw),
                  total=len(map_func_args[0]), **kw):
        pass


def numpy_stub_from_rosetta_stub(rosstub):
    npstub = np.zeros((4, 4))
    for i in range(3):
        npstub[..., i, 3] = rosstub.v[i]
        for j in range(3):
            npstub[..., i, j] = rosstub.M(i + 1, j + 1)
    npstub[..., 3, 3] = 1.0
    return npstub


def rosetta_stub_from_numpy_stub(npstub):
    rosstub = ros.core.kinematics.Stub()
    rosstub.M.xx = npstub[0, 0]
    rosstub.M.xy = npstub[0, 1]
    rosstub.M.xz = npstub[0, 2]
    rosstub.M.yx = npstub[1, 0]
    rosstub.M.yy = npstub[1, 1]
    rosstub.M.yz = npstub[1, 2]
    rosstub.M.zx = npstub[2, 0]
    rosstub.M.zy = npstub[2, 1]
    rosstub.M.zz = npstub[2, 2]
    rosstub.v.x = npstub[0, 3]
    rosstub.v.y = npstub[1, 3]
    rosstub.v.z = npstub[2, 3]
    return rosstub


def get_bb_stubs(pose, which_resi=None):
    'extract rif style stubs from rosetta pose'
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    npstubs = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            continue
        ros_stub = ros.core.kinematics.Stub(
            r.xyz('CA'), r.xyz('N'), r.xyz('CA'), r.xyz('C'))
        npstubs.append(numpy_stub_from_rosetta_stub(ros_stub))
    return np.stack(npstubs)


def pose_bounds(pose, lb, ub):
    if ub < 0: ub = len(pose) + 1 + ub
    if lb < 1 or ub > len(pose):
        raise ValueError('lb/ub ' + str(lb) + '/' + str(ub) +
                         ' out of bounds for pose with len '
                         + str(len(pose)))
    return lb, ub


def subpose(pose, lb, ub=-1):
    lb, ub = pose_bounds(pose, lb, ub)
    p = ros.core.pose.Pose()
    ros.core.pose.append_subpose_to_pose(p, pose, lb, ub)
    return p


def xform_pose(xform, pose, lb=1, ub=-1):
    lb, ub = pose_bounds(pose, lb, ub)
    if xform.shape != (4, 4):
        raise ValueError(
            'invalid xform, must be 4x4 homogeneous matrix, shape is: '
            + str(xform.shape))
    xform = rosetta_stub_from_numpy_stub(xform)
    ros.protocols.sic_dock.xform_pose(pose, xform, lb, ub)


def worst_CN_connect(p):
    for ir in range(1, len(p)):
        worst = 0
        if (p.residue(ir).is_protein() and
                p.residue(ir + 1).is_protein() and not (
                ros.core.pose.is_upper_terminus(p, ir) or
                ros.core.pose.is_lower_terminus(p, ir + 1))):
            dist = p.residue(ir).xyz('C').distance(p.residue(ir + 1).xyz('N'))
            worst = max(abs(dist - 1.32), worst)
    return worst


def no_overlapping_adjacent_residues(p):
    for ir in range(1, len(p)):
        if (p.residue(ir).is_protein() and p.residue(ir + 1).is_protein()):
            dist = p.residue(ir).xyz('CA').distance(
                p.residue(ir + 1).xyz('CA'))
            if dist < 0.1: return False
    return True


def no_overlapping_residues(p):
    for ir in range(1, len(p) + 1):
        if not p.residue(ir).is_protein():
            continue
        for jr in range(1, ir):
            if not p.residue(jr).is_protein():
                continue
            dist = p.residue(ir).xyz('CA').distance(
                p.residue(jr).xyz('CA'))
            if dist < 0.5: return False
    return True


def trim_pose(pose, resid, direction, pad=0):
    "trim end of pose from direction, leaving <=pad residues beyond resid"
    if direction not in "NC":
        raise ValueError("direction must be 'N' or 'C'")
    if not 0 < resid <= len(pose):
        raise ValueError("resid %i out of bounds %i" % (resid, len(pose)))
    p = ros.core.pose.Pose()
    if direction == 'N':
        lb, ub = max(resid - pad, 1), len(pose)
    elif direction == 'C':
        lb, ub = 1, min(resid + pad, len(pose))
    # print('_trim_pose lbub', lb, ub, 'len', len(pose), 'resid', resid)
    ros.core.pose.append_subpose_to_pose(p, pose, lb, ub)
    return p, lb, ub


def symfile_path(name):
    path, _ = os.path.split(__file__)
    return os.path.join(path, 'rosetta_symdef', name + '.sym')


@ft.lru_cache()
def get_symdata(name):
    if name is None: return None
    d = ros.core.conformation.symmetry.SymmData()
    d.read_symmetry_data_from_file(symfile_path(name))
    return d


def infer_cyclic_symmetry(pose):
    raise NotImplementedError


def bigprod(iterable):
    return ft.reduce(operator.mul, iterable, 1)


class MultiRange:

    def __init__(self, nside):
        self.nside = np.array(nside, dtype='i')
        self.psum = np.concatenate(
            [np.cumprod(self.nside[1:][::-1])[::-1], [1]])
        assert np.all(self.psum > 0)
        assert bigprod(self.nside[1:]) < 2**63
        self.len = bigprod(self.nside)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return (self[i] for i in range(self.len)[idx])
        if idx >= self.len:
            raise StopIteration
        return tuple((idx // self.psum) % self.nside)

    def __len__(self):
        return self.len
