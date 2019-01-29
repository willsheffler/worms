from .base import *
from worms import util
from worms.criteria import make_hash_table
from homog import numba_axis_angle, hrot
from xbin import gu_xbin_indexer, numba_xbin_indexer
from copy import deepcopy
from worms.util import ros


class Null(WormCriteria):
    def __init__(self, from_seg=0, origin_seg=None, to_seg=-1):
        self.from_seg = from_seg
        self.to_seg = to_seg
        self.origin_seg = origin_seg

    def score(self, segpos, *, verbosity=False, **kw):
        return 0.0

    def alignment(self, segpos, **kw):
        return np.eye(4)

    def jit_lossfunc(self):
        @util.jit
        def func(pos, idx, verts):
            return 0.0

        return func

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
        return [(self, bbs)], None

    def merge_segment(self, **kw):
        return self.from_seg

    def cloned_segments(self):
        "which bbs are being merged together"
        return self.from_seg, self.to_seg

    def iface_rms(self, pose0, prov, **kw):
        return -1
