from .base import WormCriteria

import numpy as np

from worms.util import jit


class Stack(WormCriteria):
    """
    """

    def __init__(self, sym, *, from_seg=0, tolerance=1.0, lever=50, to_seg=-1):
        if from_seg == to_seg:
            raise ValueError('from_seg should not be same as to_seg')
        self.sym = sym
        self.from_seg = from_seg
        self.tolerance = tolerance
        self.lever = lever
        self.to_seg = to_seg
        self.rot_tol = tolerance / lever
        self.is_cyclic = False
        self.symname = 'C' + str(self.sym)
        self.origin_seg = None

    def score(self):
        raise NotImplementedError

    def jit_lossfunc(self):
        from_seg = self.from_seg
        to_seg = self.to_seg
        tol2 = self.tolerance**2
        rot_tol2 = self.rot_tol**2

        @jit
        def func(pos, idx, verts):
            cen2 = pos[to_seg, :, 3].copy()  #  this was a good bug!
            ax2 = pos[to_seg, :, 2]
            cen2[2] = 0.0
            dist2 = np.sum(cen2**2)
            ang2 = np.arccos(np.abs(ax2[2]))**2
            err = np.sqrt(ang2 / rot_tol2 + dist2 / tol2)
            return err

        return func

    def alignment(self, segpos, debug=0, **kw):
        return np.eye(4)

    def merge_segment(self, **kw):
        return None

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
        "return spearate criteria for each search stage"
        return [(self, bbs)], None

    def cloned_segments(self):
        "which bbs are being merged together"
        return (self.from_seg, )

    def iface_rms(self, pose0, prov, **kw):
        return -1
