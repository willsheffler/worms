from . import WormCriteria
import numpy as np
import homog as hm
from worms.util import jit


class DihedralLattice(WormCriteria):
    """
    cases
    D3 C3

    D2 C4
    D4 C2
    D4 C4

    D2 C3
    D2 C6
    D3 C2
    D3 C6
    D6 C2
    D6 C3
    """

    def __init__(
        self,
        symname,
        from_seg,
        d_nfold,
        c_nfold,
        *,
        tolerance=1.0,
        lever=50,
        to_seg=-1,
        space_group_str=None,
    ):
        self.symname = symname
        self.from_seg = from_seg
        assert d_nfold, c_nfold in (
            (3, 3),  # P3m
            (2, 4),  # P4m
            (4, 2),
            (4, 4),
            (2, 3),  # P6m
            (2, 6),
            (3, 2),
            (3, 6),
            (6, 2),
            (6, 3),
        )
        self.d_nfold = d_nfold
        self.c_nfold = c_nfold
        self.d_2folds = get_2folds(d_nfold)
        self.tolerance = tolerance
        self.lever = lever
        self.to_seg = to_seg
        self.space_group_str = space_group_str
        self.is_cyclic = False
        self.origin_seg = None

        print(f"DihedralLattice D{d_nfold} component 2folds:")
        for x in self.d_2folds:
            print("   ", x, hm.angle_degrees(x, self.d_2folds[0]))

    def score(self, segpos, **kw):
        raise NotImplementedError

    def jit_lossfunc(self):
        d_nfold = self.d_nfold
        c_nfold = self.c_nfold
        to_seg = self.to_seg
        tolerance = self.tolerance
        lever = self.lever
        d_2folds = self.d_2folds
        origin = np.array([0, 0, 0])

        @jit
        def lossfunc_Dx_Cx(pos, idx, verts):
            ax = pos[to_seg, :3, 2]
            cn = pos[to_seg, :3, 3]
            mn = 9e9
            for i in range(d_2folds.shape[0]):
                d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
                mn = min(d, mn)
            carterr2 = d ** 2
            angerr2 = (np.arccos(abs(ax[2])) * lever) ** 2
            return np.sqrt(carterr2 + angerr2)

        @jit
        def lossfunc_Dx_C2(pos, idx, verts):
            ax = pos[to_seg, :3, 2]
            cn = pos[to_seg, :3, 3]

            if abs(ax[2]) > 0.5:
                # case: c2 on z
                mn = 9e9
                for i in range(d_2folds.shape[0]):
                    d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
                    mn = min(d, mn)
                carterr2 = d ** 2
                angerr2 = (np.arccos(abs(ax[2])) * lever) ** 2
                return np.sqrt(carterr2 + angerr2)
            else:
                # case: C2 in plane of sym and perp to main D2 axis
                dot = 9e9
                for i in range(d_2folds.shape[0]):
                    dot = min(dot, abs(np.sum(d_2folds[i] * ax)))
                angerr2_a = (np.arcsin(dot) * lever) ** 2
                angerr2_b = (np.arcsin(ax[2]) * lever) ** 2
                carterr2 = cn[2] ** 2
                return np.sqrt(angerr2_a + angerr2_b + carterr2)

        @jit
        def lossfunc_D2_Cx(pos, idx, verts):
            ax = pos[to_seg, :3, 2]
            cn = pos[to_seg, :3, 3]
            mn, mni = 9e9, -1
            mxdot, mxdoti = 0, -1
            for i in range(d_2folds.shape[0]):
                d = hm.numba_line_line_distance_pa(cn, ax, origin, d_2folds[i])
                if d < mn:
                    mn, mni = d, i
                dot = abs(np.sum(d_2folds[i] * ax))
                if dot > mxdot:
                    mxdot, mxdoti = dot, i
            if mxdoti == mni:
                # intersect same axis as parallel to, bad
                return 9e9
            carterr2 = d ** 2
            angerr2 = (np.arccos(mxdot) * lever) ** 2
            return np.sqrt(carterr2 + angerr2)

        if d_nfold > 2 and c_nfold > 2:
            return lossfunc_Dx_Cx
        if d_nfold > 2 and c_nfold == 2:
            return lossfunc_Dx_C2
        if d_nfold == 2 and c_nfold > 2:
            return lossfunc_D2_Cx

        raise NotImplementedError

    def alignment(self, segpos, out_cell_spacing=False, **kw):
        ax = segpos[self.to_seg, :3, 2]
        cn = segpos[self.to_seg, :3, 3]

        if self.d_nfold > 2 and self.c_nfold > 2:
            mn, mni = 9e9, -1
            for i, tf in enumerate(self.d_2folds):
                d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
                if d < mn:
                    mn, mni = d, i
            p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
            spacing = np.linalg.norm(p + q) / 2
            xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])
        elif self.d_nfold > 2 and self.c_nfold == 2:

            if abs(ax[2]) > 0.5:
                # case: c2 on z pick d2 isects axis
                mn, mni = 9e9, -1
                for i, tf in enumerate(self.d_2folds):
                    d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
                    if d < mn:
                        mn, mni = d, i
                p, q = hm.line_line_closest_points_pa(
                    cn, ax, [0, 0, 0], self.d_2folds[mni]
                )
                spacing = np.linalg.norm(p + q) / 2
                xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])
            else:
                # case: c2 prep to z, pick D2 perp to axis
                mn, mni = 9e9, -1
                for i, tf in enumerate(self.d_2folds):
                    d = abs(np.sum(tf * ax))
                    if d < mn:
                        mn, mni = d, i
                p, q = hm.line_line_closest_points_pa(
                    cn, ax, [0, 0, 0], self.d_2folds[mni]
                )
                spacing = np.linalg.norm(p + q) / 2
                xalign = hm.align_vectors([0, 0, 1], q, [0, 0, 1], [1, 0, 0])

        elif self.d_nfold == 2 and self.c_nfold > 2:
            mn, mni = 9e9, -1
            mxdot, mxdoti = 0, -1
            for i, tf in enumerate(self.d_2folds):
                d = hm.line_line_distance_pa(cn, ax, [0, 0, 0], tf)
                if d < mn:
                    mn, mni = d, i
                dot = abs(np.sum(tf * ax))
                if dot > mxdot:
                    mxdot, mxdoti = dot, i
            assert mni != mxdoti
            p, q = hm.line_line_closest_points_pa(cn, ax, [0, 0, 0], self.d_2folds[mni])
            spacing = np.linalg.norm(p + q) / 2
            # assumes d_folds are X Y Z ax[argmin] selects correct one
            xalign = hm.align_vectors(
                self.d_2folds[mni], self.d_2folds[mxdoti], [1, 0, 0], [0, 0, 1]
            )
            # print("cn", cn)
            # print("ax", ax)
            # print("isect", self.d_2folds[mni])
            # print("align", self.d_2folds[mxdoti])
            # print("align xform", xalign)
            # print("aligned ax", xalign[:3, :3] @ ax)
            # assert 0

        else:
            raise NotImplementedError

        if out_cell_spacing:
            return xalign, spacing
        else:
            return xalign

    def symfile_modifiers(self, segpos):
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        return dict(scale_positions=cell_dist)

    def merge_segment(self, **kw):
        return self.from_seg

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
        "return spearate criteria for each search stage"
        return [(self, bbs)], None

    def cloned_segments(self):
        "which bbs are being merged together"
        return (self.from_seg,)

    def iface_rms(self, pose0, prov, **kw):
        return -1


def get_2folds(n):
    if n is 2:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if n is 3:
        return np.array(
            [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]]
        )
    if n is 4:
        return np.array(
            [
                [1, 0, 0],
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [0, 1, 0],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            ]
        )
    assert 0, ""


def P3m_D3_C3(d3=0, c3=-1):
    return DihedralLattice("P3m_D3_C3_3", d_nfold=3, c_nfold=3, from_seg=d3, to_seg=c3)


def P4m_D2_C4(d2=0, c4=-1):
    return DihedralLattice("P4m_D2_C4_4", d_nfold=2, c_nfold=4, from_seg=d2, to_seg=c4)


def P4m_D4_C2(d4=0, c2=-1):
    return DihedralLattice("P4m_D4_C2_3", d_nfold=4, c_nfold=2, from_seg=d4, to_seg=c2)


def P4m_D4_C4(d4=0, c4=-1):
    return DihedralLattice("P4m_D4_C4_3", d_nfold=4, c_nfold=4, from_seg=d4, to_seg=c4)


def P6m_D2_C3(d2=0, c3=-1):
    return DihedralLattice("P6m_D2_C3_4", d_nfold=2, c_nfold=3, from_seg=d2, to_seg=c3)


def P6m_D2_C6(d2=0, c6=-1):
    return DihedralLattice("P6m_D2_C6_3", d_nfold=2, c_nfold=6, from_seg=d2, to_seg=c6)


def P6m_D3_C2(d3=0, c2=-1):
    return DihedralLattice("P6m_D3_C2_4", d_nfold=3, c_nfold=2, from_seg=d3, to_seg=c2)


def P6m_D3_C6(d3=0, c6=-1):
    return DihedralLattice("P6m_D3_C6_3", d_nfold=3, c_nfold=6, from_seg=d3, to_seg=c6)
