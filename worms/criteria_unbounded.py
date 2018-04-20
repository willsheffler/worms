from .criteria import WormCriteria, Ux, Uz
import numpy as np
import homog as hm ## python library that Will wrote to do geometry things

class AxesAngle(WormCriteria): ## for 2D arrays (maybe 3D in the future?)
    def __init__(self, symname, tgtaxis1, tgtaxis2, from_seg, *, tol=1.0,
                 lever=50, to_seg=-1):
        """ Worms criteria for non-intersecting axes re: unbounded things

        Args:
            symname (str): Symmetry identifier, to label stuff and look up the symdef file.
            tgtaxis1: Target axis 1.
            tgtaxis2: Target axis 2. 
            from_seg (int): The segment # to start at.
            tol (float): A geometry/alignment error threshold. Vaguely Angstroms.
            lever (float): Tradeoff with distances and angles for a lever-like object. To convert an angle error to a distance error for an oblong shape.
            to_seg (int): The segment # to end at.

        """

        self.symname = symname
        self.tgtaxis1 = tgtaxis1 ## we are treating these as vectors for now
        self.tgtaxis2 = tgtaxis2
        self.from_seg = from_seg
        self.tol = tol
        self.lever = lever
        self.to_seg = to_seg
        ## if you want to store arguments, you have to write these self.argument lines

        self.target_angle = np.arccos(np.abs(hm.hdot(tgtaxis1, tgtaxis2))) ## already set to a non- self.argument in this function

    def score(self, segpos, **kw):
        """ Score

        Args:
            segpos (lst): List of segment positions / coordinates.
            **kw I'll accept any "non-positional" argument as name = value, and store in a dictionary

        """
        ## numpy arrays of how many things you are scoring, and a 4x4 translation/rotation matrix
        ax1 = segpos[self.from_seg][..., :, 2] ## from the first however many dimensions except the last two, give me the 2nd column, which for us is the Z-axis
        ax2 = segpos[self.to_seg][..., :, 2]
        #angle = hm.angle(ax1, ax2) ## homog angle function will compute the angle between two vectors, and give back an angle in radians
        angle = np.arccos(np.abs(hm.hdot(ax1, ax2))) ## this is better because it contains absolutel value, which ensures that you always get the smaller of the angles resulting from intersecting two lines
        return np.abs((angle - self.target_angle)) / self.tol * self.lever ## as tolerance goes up, you care about the angle error less. as lever goes up, you care about the angle error more. 

    def alignment(self, segpos, **kw):
        """ Alignment to move stuff to be in line with symdef file

        Args:
            segpos (lst): List of segment positions / coordinates.
            **kw I'll accept any "non-positional" argument as name = value, and store in a dictionary

        """
        cen1 = segpos[self.from_seg][..., :, 3] ## 4th column is x,y,z translation
        cen2 = segpos[self.to_seg][..., :, 3]
        ax1 = segpos[self.from_seg][..., :, 2] ## 3rd column is axis
        ax2 = segpos[self.to_seg][..., :, 2]  
        if hm.angle(ax1, ax2) > np.pi / 2: ## make sure to align with smaller axis choice
            ax2 = -ax2
        if abs(hm.angle(self.tgtaxis1, self.tgtaxis2)) < 0.1 :
            d = hm.proj_perp(ax1, cen2 - cen1) #vector delta between cen2 and cen1
            x = hm.align_vectors(ax1, d, self.tgtaxis1, [0,1,0,0] ) #align d to Y axis
            x[..., :, 3] = - x @ cen1
        else:
            x = hm.align_vectors(ax1, ax2, self.tgtaxis1, self.tgtaxis2) ## utility function that tries to align ax1 and ax2 to the target axes
            x[..., :, 3] = - x @ cen1 ## move from_seg cen1 to origin
            x[..., 2, 3] = - (x @ cen2)[...,2] ##move cen2 down along z, such that it's centered at z=0
        return x

    def symfile_modifiers(self, segpos):
        x = self.alignment(segpos)
        cen2 = segpos[self.to_seg][..., :, 3]
        cell_spacing = (x @ cen2)[..., 1] ## setting cell_spacing equal to offset of dimer from origin
        return dict(scale_positions = cell_spacing)

def Sheet_P321(c3=None, c2=None, **kw):
    if c3 is None or c2 is None:
        raise ValueError('must specify ...?') #one or two of c3, c2
    return AxesAngle('Sheet_P321_C3_C2_depth3_1comp', Uz, Ux, from_seg=c3, to_seg=c2, **kw) ##this is currently identical to the D3 format...how do we change it to make it an array?

def Sheet_P4212(c4=None, c2=None, **kw): ##should there be options for multiple C2's?
    if c4 is None or c2 is None:
        raise ValueError('must specify ...?') #one or two of c4, c2
    return AxesAngle('Sheet_P4212_C4_C2_depth3_1comp', Uz, Ux, from_seg=c4, to_seg=c2, **kw)

def Sheet_P6(c6=None, c2=None, **kw): ##should there be options for multiple C2's?
    if c6 is None or c2 is None:
        raise ValueError('must specify ...?') #one or two of c6, c2
    return AxesAngle('Sheet_P6_C6_C2_depth3_1comp', Uz, Uz, from_seg=c6, to_seg=c2, **kw)

##def Crystal_example():