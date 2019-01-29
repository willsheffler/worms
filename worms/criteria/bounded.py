from .base import WormCriteria, Ux, Uy, Uz
import numpy as np
import homog as hm
from worms.util import jit
from worms.merge.wye import wye_merge


@jit
def numba_line_line_closest_points_pa(pt1, ax1, pt2, ax2):
    C21 = pt2 - pt1
    M = hm.numba_cross(ax1, ax2)
    m2 = np.sum(M ** 2)
    R = hm.numba_cross(C21, M / m2)
    t1 = np.sum(R * ax2)
    t2 = np.sum(R * ax1)
    Q1 = pt1 - t1 * ax1
    Q2 = pt2 - t2 * ax2
    return Q1, Q2


@jit
def numba_normalized(v):
    return v / hm.numba_norm(v)


class AxesIntersect(WormCriteria):
    """
    """

    def __init__(
        self,
        symname,
        tgtaxis1,
        tgtaxis2,
        from_seg=0,
        origin_seg=None,
        *,
        tolerance=1.0,
        lever=50,
        to_seg=-1,
        distinct_axes=False,
        segs=None,
        tgtaxis3=None
    ):
        """
        """
        if from_seg == to_seg:
            raise ValueError("from_seg should not be same as to_seg")
        self.symname = symname
        self.from_seg = from_seg
        if len(tgtaxis1) == 2:
            tgtaxis1 += ([0, 0, 0, 1],)
        if len(tgtaxis2) == 2:
            tgtaxis2 += ([0, 0, 0, 1],)
        self.tgtaxis1 = (
            tgtaxis1[0],
            hm.hnormalized(tgtaxis1[1]),
            hm.hpoint(tgtaxis1[2]),
        )
        self.tgtaxis2 = (
            tgtaxis2[0],
            hm.hnormalized(tgtaxis2[1]),
            hm.hpoint(tgtaxis2[2]),
        )
        if tgtaxis3:
            if len(tgtaxis3) == 2:
                tgtaxis3 += ([0, 0, 0, 1],)
            self.tgtaxis3 = (
                tgtaxis3[0],
                hm.hnormalized(tgtaxis3[1]),
                hm.hpoint(tgtaxis3[2]),
            )
            assert 3 == len(self.tgtaxis3)
        assert 3 == len(self.tgtaxis1)
        assert 3 == len(self.tgtaxis2)

        self.tgtangle = hm.angle(tgtaxis1[1], tgtaxis2[1])
        self.tolerance = tolerance
        self.lever = lever
        self.to_seg = to_seg
        self.rot_tol = tolerance / lever
        self.distinct_axes = distinct_axes  # -z not same as z (for T33)
        self.sym_axes = [self.tgtaxis1, self.tgtaxis2]
        self.is_cyclic = False
        self.origin_seg = None
        self.segs = segs

    def score(self, segpos, verbosity=False, **kw):
        cen1 = segpos[self.from_seg][..., :, 3]
        cen2 = segpos[self.to_seg][..., :, 3]
        ax1 = segpos[self.from_seg][..., :, 2]
        ax2 = segpos[self.to_seg][..., :, 2]
        if self.distinct_axes:
            p, q = hm.line_line_closest_points_pa(cen1, ax1, cen2, ax2)
            dist = hm.hnorm(p - q)
            cen = (p + q) / 2
            ax1c = hm.hnormalized(cen1 - cen)
            ax2c = hm.hnormalized(cen2 - cen)
            ax1 = np.where(hm.hdot(ax1, ax1c)[..., None] > 0, ax1, -ax1)
            ax2 = np.where(hm.hdot(ax2, ax2c)[..., None] > 0, ax2, -ax2)
            ang = np.arccos(hm.hdot(ax1, ax2))
        else:
            dist = hm.line_line_distance_pa(cen1, ax1, cen2, ax2)
            ang = np.arccos(np.abs(hm.hdot(ax1, ax2)))
        roterr2 = (ang - self.tgtangle) ** 2
        return np.sqrt(roterr2 / self.rot_tol ** 2 + (dist / self.tolerance) ** 2)

    def jit_lossfunc(self):
        from_seg = self.from_seg
        to_seg = self.to_seg
        tgtangle = self.tgtangle
        tolerance = self.tolerance
        rot_tol = self.rot_tol
        distinct_axes = self.distinct_axes

        @jit
        def func(pos, idx, verts):
            cen1 = pos[from_seg][:, 3]
            cen2 = pos[to_seg][:, 3]
            ax1 = pos[from_seg][:, 2]
            ax2 = pos[to_seg][:, 2]
            if distinct_axes:
                cen1 = cen1[:3]
                cen2 = cen2[:3]
                ax1 = ax1[:3]
                ax2 = ax2[:3]
                p, q = numba_line_line_closest_points_pa(cen1, ax1, cen2, ax2)
                dist = np.sqrt(np.sum((p - q) ** 2))
                cen = (p + q) / 2
                ax1c = numba_normalized(cen1 - cen)
                ax2c = numba_normalized(cen2 - cen)
                if np.sum(ax1 * ax1c) < 0:
                    ax1 = -ax1
                if np.sum(ax2 * ax2c) < 0:
                    ax2 = -ax2
                ang = np.arccos(np.sum(ax1 * ax2))
            else:
                dist = hm.numba_line_line_distance_pa(cen1, ax1, cen2, ax2)
                ang = np.arccos(np.abs(hm.numba_dot(ax1, ax2)))
            roterr2 = (ang - tgtangle) ** 2
            return np.sqrt(roterr2 / rot_tol ** 2 + (dist / tolerance) ** 2)

        return func

    def alignment(self, segpos, debug=0, **kw):
        """
        """
        if hm.angle_degrees(self.tgtaxis1[1], self.tgtaxis2[1]) < 0.1:
            return np.eye(4)
        cen1 = segpos[self.from_seg][..., :, 3]
        cen2 = segpos[self.to_seg][..., :, 3]
        ax1 = segpos[self.from_seg][..., :, 2]
        ax2 = segpos[self.to_seg][..., :, 2]
        if not self.distinct_axes and hm.angle(ax1, ax2) > np.pi / 2:
            ax2 = -ax2
        p, q = hm.line_line_closest_points_pa(cen1, ax1, cen2, ax2)
        cen = (p + q) / 2
        # ax1 = hm.hnormalized(cen1 - cen)
        # ax2 = hm.hnormalized(cen2 - cen)
        x = hm.align_vectors(ax1, ax2, self.tgtaxis1[1], self.tgtaxis2[1])
        x[..., :, 3] = -x @ cen
        if debug:
            print(
                "angs",
                hm.angle_degrees(ax1, ax2),
                hm.angle_degrees(self.tgtaxis1[1], self.tgtaxis2[1]),
            )
            print("ax1", ax1)
            print("ax2", ax2)
            print("xax1", x @ ax1)
            print("tax1", self.tgtaxis1[1])
            print("xax2", x @ ax2)
            print("tax2", self.tgtaxis2[1])
            raise AssertionError
            # if not (np.allclose(x @ ax1, self.tgtaxis1[1], atol=1e-2) and
            #         np.allclose(x @ ax2, self.tgtaxis2[1], atol=1e-2)):
            #     print(hm.angle(self.tgtaxis1[1], self.tgtaxis2[1]))
            #     print(hm.angle(ax1, ax2))
            #     print(x @ ax1)
            #     print(self.tgtaxis1[1])
            #     print(x @ ax2)
            #     print(self.tgtaxis2[1])
            #     raise AssertionError('hm.align_vectors sucks')

        return x

    def merge_segment(self, **kw):
        if self.origin_seg is None:
            return None
        return self.from_seg

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, topology, **kw):
        "return spearate criteria for each search stage"
        if topology.is_linear():
            return [(self, bbs)], None

        # 3 component cage
        paths = topology.paths()
        assert len(paths) == 2
        segmap = {s: i for i, s in enumerate(self.segs)}
        axes = [self.tgtaxis1, self.tgtaxis2, self.tgtaxis3]
        from_seg = paths[0][0]
        to_segA = paths[0][-1]
        to_segB = paths[1][-1]

        critA = AxesIntersect(
            symname=self.symname,
            from_seg=from_seg,
            to_seg=to_segA,
            tgtaxis1=axes[segmap[from_seg]],
            tgtaxis2=axes[segmap[to_segA]],
        )
        critA.bbspec = [self.bbspec[i] for i in paths[0]]
        bbsA = [bbs[i] for i in paths[0]]

        critB = AxesIntersect(
            symname=self.symname,
            from_seg=from_seg,
            to_seg=to_segB,
            tgtaxis1=axes[segmap[from_seg]],
            tgtaxis2=axes[segmap[to_segB]],
        )
        critB.bbspec = [self.bbspec[i] for i in paths[1]]
        bbsB = [bbs[i] for i in paths[1]]

        print("3 comp cage stages!")

        return [(critA, bbsA), (critB, bbsB)], wye_merge

    # def cloned_segments(self):
    # "which bbs are being merged together"
    # return (self.from_seg, )

    def iface_rms(self, pose0, prov, **kw):
        return -1


def D2(c2=0, c2b=-1, **kw):
    return AxesIntersect("D2", (2, Uz), (2, Ux), c2, to_seg=c2b, **kw)


def D3(c3=0, c2=-1, **kw):
    return AxesIntersect("D3", (3, Uz), (2, Ux), c3, to_seg=c2, **kw)


def D4(c4=0, c2=-1, **kw):
    return AxesIntersect("D4", (4, Uz), (2, Ux), c4, to_seg=c2, **kw)


def D5(c5=0, c2=-1, **kw):
    return AxesIntersect("D5", (5, Uz), (2, Ux), c5, to_seg=c2, **kw)


def D6(c6=0, c2=-1, **kw):
    return AxesIntersect("D6", (6, Uz), (2, Ux), c6, to_seg=c2, **kw)


def Tetrahedral(c3=None, c2=None, c3b=None, **kw):
    if 1 is not (c3b is None) + (c3 is None) + (c2 is None):
        raise ValueError("must specify exactly two of c3, c2, c3b")
    if c2 is None:
        from_seg, to_seg, nf1, nf2, ex = c3b, c3, 7, 3, 2
    if c3 is None:
        from_seg, to_seg, nf1, nf2, ex = c3b, c2, 7, 2, 3
    if c3b is None:
        from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 7
    return AxesIntersect(
        "T",
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(min(3, nf1), hm.sym.tetrahedral_axes[nf1]),
        tgtaxis2=(min(3, nf2), hm.sym.tetrahedral_axes[nf2]),
        distinct_axes=(nf1 == 7),
        **kw
    )


def Octahedral(c4=None, c3=None, c2=None, **kw):
    if c4 is not None and c3 is not None and c2 is not None:
        return AxesIntersect(
            "O",
            segs=(c2, c3, c4),
            tgtaxis1=(2, hm.sym.octahedral_axes[2]),
            tgtaxis2=(3, hm.sym.octahedral_axes[3]),
            tgtaxis3=(4, hm.sym.octahedral_axes[4]),
        )
    if 1 is not (c4 is None) + (c3 is None) + (c2 is None):
        raise ValueError("must specify exactly two of c4, c3, c2")
    if c2 is None:
        from_seg, to_seg, nf1, nf2, ex = c4, c3, 4, 3, 2
    if c3 is None:
        from_seg, to_seg, nf1, nf2, ex = c4, c2, 4, 2, 3
    if c4 is None:
        from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 4
    return AxesIntersect(
        "O",
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(nf1, hm.sym.octahedral_axes[nf1]),
        tgtaxis2=(nf2, hm.sym.octahedral_axes[nf2]),
        **kw
    )


def Icosahedral(c5=None, c3=None, c2=None, **kw):
    if 1 is not (c5 is None) + (c3 is None) + (c2 is None):
        raise ValueError("must specify exactly two of c5, c3, c2")
    if c2 is None:
        from_seg, to_seg, nf1, nf2, ex = c5, c3, 5, 3, 2
    if c3 is None:
        from_seg, to_seg, nf1, nf2, ex = c5, c2, 5, 2, 3
    if c5 is None:
        from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 5
    return AxesIntersect(
        "I",
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(nf1, hm.sym.icosahedral_axes[nf1]),
        tgtaxis2=(nf2, hm.sym.icosahedral_axes[nf2]),
        **kw
    )
