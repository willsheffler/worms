from . import WormCriteria, Ux, Uz
import numpy as np
import homog as hm  ## python library that Will wrote to do geometry things
from worms.util import jit


class AxesAngle(WormCriteria):  ## for 2D arrays (maybe 3D in the future?)
    def __init__(
            self,
            symname,
            tgtaxis1,
            tgtaxis2,
            from_seg,
            *,
            tolerance=1.0,
            lever=50,
            to_seg=-1,
            space_group_str=None,
            cell_dist_scale=1.0
    ):
        """ Worms criteria for non-intersecting axes re: unbounded things

        Args:
            symname (str): Symmetry identifier, to label stuff and look up the symdef file.
            tgtaxis1: Target axis 1.
            tgtaxis2: Target axis 2.
            from_seg (int): The segment # to start at.
            tolerance (float): A geometry/alignment error threshold. Vaguely Angstroms.
            lever (float): Tradeoff with distances and angles for a lever-like object. To convert an angle error to a distance error for an oblong shape.
            to_seg (int): The segment # to end at.
            space_group_str: The target space group.

        """

        self.symname = symname
        self.cell_dist_scale = cell_dist_scale
        self.tgtaxis1 = np.asarray(
            tgtaxis1, dtype='f8'
        )  ## we are treating these as vectors for now, make it an array if it isn't yet, set array type to 8-type float
        self.tgtaxis2 = np.asarray(tgtaxis2, dtype='f8')
        # print(self.tgtaxis1.shape)
        # print(np.linalg.norm(tgtaxis1))
        self.tgtaxis1 /= np.linalg.norm(
            self.tgtaxis1
        )  #normalize target axes to 1,1,1
        self.tgtaxis2 /= np.linalg.norm(self.tgtaxis2)
        if hm.angle(self.tgtaxis1, self.tgtaxis2) > np.pi / 2:
            self.tgtaxis2 = -self.tgtaxis2
        self.from_seg = from_seg
        self.tolerance = tolerance
        self.lever = lever
        self.to_seg = to_seg
        self.space_group_str = space_group_str
        ## if you want to store arguments, you have to write these self.argument lines

        self.target_angle = np.arccos(
            np.abs(hm.hdot(self.tgtaxis1, self.tgtaxis2))
        )  ## already set to a non- self.argument in this function
        # print(self.target_angle * (180 / np.pi))
        self.is_cyclic = False
        self.origin_seg = None

    def score(self, segpos, **kw):
        ax1 = segpos[self.from_seg][..., :, 2]
        ax2 = segpos[self.to_seg][..., :, 2]
        angle = np.arccos(np.abs(hm.hdot(ax1, ax2)))
        return np.abs((angle - self.target_angle)
                      ) / self.tolerance * self.lever

    def jit_lossfunc(self):
        from_seg = self.from_seg
        to_seg = self.to_seg
        target_angle = self.target_angle
        tolerance = self.tolerance
        lever = self.lever

        @jit
        def func(pos, idx, verts):
            ax1 = pos[from_seg, :3, 2]
            ax2 = pos[to_seg, :3, 2]
            angle = np.arccos(np.abs(np.sum(ax1 * ax2)))
            return np.abs((angle - target_angle)) / tolerance * lever

        return func

    def alignment(self, segpos, out_cell_spacing=False, **kw):
        """ Alignment to move stuff to be in line with symdef file

        Args:
            segpos (lst): List of segment positions / coordinates.
            **kw I'll accept any "non-positional" argument as name = value, and store in a dictionary

        """
        cen1 = segpos[self.from_seg][..., :, 3]
        cen2 = segpos[self.to_seg][..., :, 3]
        ax1 = segpos[self.from_seg][..., :, 2]  ## 3rd column is axis
        ax2 = segpos[self.to_seg][..., :, 2]
        ## make sure to align with smaller axis choice
        if hm.angle(ax1, ax2) > np.pi / 2:
            ax2 = -ax2
        if abs(hm.angle(self.tgtaxis1, self.tgtaxis2)) < 0.1:
            #vector delta between cen2 and cen1
            d = hm.proj_perp(ax1, cen2 - cen1)
            Xalign = hm.align_vectors(ax1, d, self.tgtaxis1,
                                      [0, 1, 0, 0])  #align d to Y axis
            Xalign[..., :, 3] = -Xalign @ cen1
            cell_dist = (Xalign @ cen2)[..., 1]
        else:
            try:
                Xalign = hm.align_vectors(
                    ax1, ax2, self.tgtaxis1, self.tgtaxis2
                )
            except AssertionError as e:
                print('align_vectors error')
                print('   ', ax1)
                print('   ', ax2)
                print('   ', self.tgtaxis1)
                print('   ', self.tgtaxis2)
                raise e
            Xalign[..., :, 3] = -Xalign @ cen1  ## move from_seg cen1 to origin
            cen2_0 = Xalign @ cen2  #moving cen2 by Xalign
            D = np.stack([self.tgtaxis1[:3], [0, 1, 0], self.tgtaxis2[:3]]).T
            #matrix where the columns are the things in the list
            #CHANGE Uy to an ARGUMENT SOON!!!!
            #print("D: ", D)
            A1offset, cell_dist, _ = np.linalg.inv(D) @ cen2_0[:3]
            #transform of A1 offest, cell distance (offset along other axis), and A2 offset (<-- we are ignoring this)
            Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * self.tgtaxis1)
            #Xalign[..., :, 3] = Xalign[..., :, 3] + [0,cell_dist,0,0]
        if out_cell_spacing:
            return Xalign, cell_dist
        else:
            return Xalign

    def symfile_modifiers(self, segpos):
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        return dict(scale_positions=cell_dist * self.cell_dist_scale)

    def crystinfo(self, segpos):
        #CRYST1   85.001   85.001   85.001  90.00  90.00  90.00 P 21 3
        if self.space_group_str is None:
            return None
        #print("hi")
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        cell_dist = abs(cell_dist * self.cell_dist_scale)
        return cell_dist, cell_dist, cell_dist, 90, 90, 90, self.space_group_str

    def merge_segment(self, **kw):
        return self.from_seg

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
        "return spearate criteria for each search stage"
        return [(self, bbs)], None

    def cloned_segments(self):
        "which bbs are being merged together"
        return (self.from_seg, )

    def iface_rms(self, pose0, prov, **kw):
        return -1


def Sheet_P321(c3=None, c2=None, **kw):
    if c3 is None or c2 is None:
        raise ValueError('must specify ...?')  #one or two of c3, c2
    return AxesAngle(
        'Sheet_P321_C3_C2_depth3_1comp', Uz, Ux, from_seg=c3, to_seg=c2, **kw
    )  ##this is currently identical to the D3 format...how do we change it to make it an array?


def Sheet_P4212(c4=None, c2=None, **kw):
    ##should there be options for multiple C2's?
    if c4 is None or c2 is None:
        raise ValueError('must specify ...?')  #one or two of c4, c2
    return AxesAngle(
        'Sheet_P4212_C4_C2_depth3_1comp', Uz, Ux, from_seg=c4, to_seg=c2, **kw
    )


def Sheet_P6(c6=None, c2=None, **kw):
    if c6 is None or c2 is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    return AxesAngle(
        'Sheet_P6_C6_C2_depth3_1comp', Uz, Uz, from_seg=c6, to_seg=c2, **kw
    )


#### WORKING ####
def Crystal_P213_C3_C3(c3a=None, c3b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_P213_C3_C3_depth3_1comp',
        [1, 1, 1, 0],
        [-1, -1, 1, 0],
        from_seg=c3a,
        to_seg=c3b,
        space_group_str="P 21 3",
        cell_dist_scale=2.0,  # for some reason, this one needs this
        **kw
    )
    #dihedral angle = 70.5288


#### IN PROGRESS ####
# I just normalized all the angles, but I don't think you can do this...might need to check the angle between them. Print and check that it is correct.
def Crystal_P4132_C2_C3(c2a=None, c3b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_P4132_C2_C3_depth3_1comp', [0, -1, 1, 0], [-1, -1, 0, 0],
        from_seg=c2a,
        to_seg=c3b,
        space_group_str="P 41 3 2",
        **kw
    )
    #dihedral angle = 35.2644


def Crystal_I213_C2_C3(c2a=None, c3b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_I213_C2_C3_depth3_1comp', [0, 0, 1, 0], [-1, 1, 1, 0],
        from_seg=c2a,
        to_seg=c3b,
        space_group_str="I 21 3",
        **kw
    )
    #dihedral angle = 54.7356


def Crystal_I432_C2_C4(c2a=None, c4b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_I432_C2_C4_depth3_1comp', [-1, 0, 1, 0], [0, 0, 1, 0],
        from_seg=c2a,
        to_seg=c4b,
        space_group_str="I 4 3 2",
        **kw
    )
    #dihedral angle = 45


def Crystal_F432_C3_C4(c3a=None, c4b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_F432_C3_C4_depth3_1comp', [-1, 1, 1, 0], [0, 1, 0, 0],
        from_seg=c3a,
        to_seg=c4b,
        space_group_str="F 4 3 2",
        **kw
    )
    #dihedral angle = 54.7356


def Crystal_P432_C4_C4(c4a=None, c4b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?')  #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle(
        'Crystal_P432_C4_C4_depth3_1comp', [0, 0, 1, 0], [0, 1, 0, 0],
        from_seg=c4a,
        to_seg=c4b,
        space_group_str="P 4 3 2",
        **kw
    )
    #dihedral angle = 90


class DihedralLattice(WormCriteria):
    def __init__(
            self,
            symname,
            tgtaxis1,
            tgtaxis2,
            from_seg,
            *,
            tolerance=1.0,
            lever=50,
            to_seg=-1,
            space_group_str=None,
            cell_dist_scale=1.0
    ):
        self.symname = symname
        self.cell_dist_scale = cell_dist_scale
        self.tgtaxis1 = np.asarray(tgtaxis1, dtype='f8')
        self.tgtaxis2 = np.asarray(tgtaxis2, dtype='f8')
        # print(self.tgtaxis1.shape)
        # print(np.linalg.norm(tgtaxis1))
        self.tgtaxis1 /= np.linalg.norm(self.tgtaxis1)
        self.tgtaxis2 /= np.linalg.norm(self.tgtaxis2)
        assert np.sum(tgtaxis1 * tgtaxis2) <= 0.0001
        self.from_seg = from_seg
        self.tolerance = tolerance
        self.lever = lever
        self.to_seg = to_seg
        self.space_group_str = space_group_str
        self.is_cyclic = False
        self.origin_seg = None

    def score(self, segpos, **kw):
        raise NotImplementedError

    def jit_lossfunc(self):
        to_seg = self.to_seg
        tgt1 = self.tgt1
        tgt2 = self.tgt2
        tolerance = self.tolerance
        lever = self.lever

        @jit
        def func(pos, idx, verts):
            ax2 = pos[to_seg, :3, 2]
            cn2 = pos[to_seg, :3, 2]
            ang2 = np.arccos(np.sum(ax2 * tgt1))**2
            cn2_T_tgt1 = cn2 - np.sum(tgt1 * cn2) / np.sum(tgt1 * tgt1) * tgt1
            angb2 = np.arccos(np.sum(cen2_T_tgt1 * tgt2))**2
            return np.sqrt(ang2 + angb2) / tolerance * lever

        return func

    def alignment(self, segpos, out_cell_spacing=False, **kw):
        return np.eye(4)

    def symfile_modifiers(self, segpos):
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        return dict(scale_positions=cell_dist * self.cell_dist_scale)

    def merge_segment(self, **kw):
        return self.from_seg

    def stages(self, hash_cart_resl, hash_ori_resl, bbs, **kw):
        "return spearate criteria for each search stage"
        return [(self, bbs)], None

    def cloned_segments(self):
        "which bbs are being merged together"
        return (self.from_seg, )

    def iface_rms(self, pose0, prov, **kw):
        return -1
