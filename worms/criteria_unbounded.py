from .criteria import WormCriteria, Ux, Uz
import numpy as np
import homog as hm ## python library that Will wrote to do geometry things
import pyrosetta

class AxesAngle(WormCriteria): ## for 2D arrays (maybe 3D in the future?)
    def __init__(self, symname, tgtaxis1, tgtaxis2, from_seg, *, tol=1.0,
                 lever=50, to_seg=-1, space_group_str=None):
        """ Worms criteria for non-intersecting axes re: unbounded things

        Args:
            symname (str): Symmetry identifier, to label stuff and look up the symdef file.
            tgtaxis1: Target axis 1.
            tgtaxis2: Target axis 2. 
            from_seg (int): The segment # to start at.
            tol (float): A geometry/alignment error threshold. Vaguely Angstroms.
            lever (float): Tradeoff with distances and angles for a lever-like object. To convert an angle error to a distance error for an oblong shape.
            to_seg (int): The segment # to end at.
            space_group_str: The target space group.

        """

        self.symname = symname
        self.tgtaxis1 = np.asarray(tgtaxis1,dtype='f8') ## we are treating these as vectors for now, make it an array if it isn't yet, set array type to 8-type float
        self.tgtaxis2 = np.asarray(tgtaxis2,dtype='f8')
        print(self.tgtaxis1.shape)
        print(np.linalg.norm(tgtaxis1))
        self.tgtaxis1 /= np.linalg.norm(self.tgtaxis1) #normalize target axes to 1,1,1
        self.tgtaxis2 /= np.linalg.norm(self.tgtaxis2)
        if hm.angle(self.tgtaxis1,self.tgtaxis2) > np.pi/2:
            self.tgtaxis2 = -self.tgtaxis2
        self.from_seg = from_seg
        self.tol = tol
        self.lever = lever
        self.to_seg = to_seg
        self.space_group_str = space_group_str
        ## if you want to store arguments, you have to write these self.argument lines

        self.target_angle = np.arccos(np.abs(hm.hdot(self.tgtaxis1, self.tgtaxis2))) ## already set to a non- self.argument in this function
        print(self.target_angle * (180/np.pi))

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

    def alignment(self, segpos, out_cell_spacing=False, **kw):
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
            Xalign = hm.align_vectors(ax1, d, self.tgtaxis1, [0,1,0,0] ) #align d to Y axis
            Xalign[..., :, 3] = - Xalign @ cen1
            cell_dist = (Xalign @ cen2)[..., 1]

        else:
            Xalign = hm.align_vectors(ax1, ax2, self.tgtaxis1, self.tgtaxis2) ## utility function that tries to align ax1 and ax2 to the target axes
            Xalign[..., :, 3] = - Xalign @ cen1 ## move from_seg cen1 to origin
            cen2_0 = Xalign@cen2 #moving cen2 by Xalign
            D = np.stack([self.tgtaxis1[:3] , [0,1,0] , self.tgtaxis2[:3]]).T #matrix where the columns are the things in the list
            #CHANGE Uy to an ARGUMENT SOON!!!!
            #print("D: ", D)
            A1offset, cell_dist, _ = np.linalg.inv(D)@cen2_0[:3] #transform of A1 offest, cell distance (offset along other axis), and A2 offset (<-- we are ignoring this)
            Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * self.tgtaxis1)
            #Xalign[..., :, 3] = Xalign[..., :, 3] + [0,cell_dist,0,0]
        if out_cell_spacing:
            #print(2*cell_dist)
            return Xalign, cell_dist
        else:
            return Xalign


            #this other attempt to do things with math
            # a = (cen2_0[2] * self.tgtaxis2[0]) - (cen2_0[0] *(self.tgtaxis2[0])) / ((self.tgtaxis2[2] * self.tgtaxis1[0]) - (self.tgtaxis1[2] * self.tgtaxis2[0]) )
            # print("a denominator: ",((self.tgtaxis2[2] * self.tgtaxis1[0]) - (self.tgtaxis1[2] * self.tgtaxis2[0]) ) )
            # c = (cen2_0[2] + (a * self.tgtaxis1[2])) / self.tgtaxis2[2]
            # b = (c * self.tgtaxis2[1]) - cen2_0[0] - (a * self.tgtaxis1[1])

            # Xalign[...,:, 3] = Xalign[...,:, 3] + (a * self.tgtaxis1)
            # print("a: ",a)
            # print("b: ",b)
            # print("c: ",c)
            # print("Xalign: ",Xalign)

            #original transformations for when center's are aligned on coordinate axes
            #Xalign[..., :, 3] = - Xalign @ cen1 ## move from_seg cen1 to origin
            #Xalign[..., 2, 3] = - (Xalign @ cen2)[...,2] ##move cen2 down along z, such that it's centered at z=0
            

            # print("ax1: ",ax1)
            # print("ax2: ",ax2)
            # print("angle between ax1 & ax2: ",np.arccos(np.abs(hm.hdot(ax1, ax2)))* (180/np.pi))
            # print("angle between tgtaxis1 & tgtaxis2: ",np.arccos(np.abs(hm.hdot(self.tgtaxis1, self.tgtaxis2)))* (180/np.pi))
            # print("tgtaxis1: ",self.tgtaxis1)
            # print("tgtaxis2: ",self.tgtaxis2)
            # print("aligned ax1: ",Xalign@ax1)
            # print("aligned ax2: ",Xalign@ax2)

    def symfile_modifiers(self, segpos):
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        return dict(scale_positions = cell_dist)

    def crystinfo(self, segpos):
        #CRYST1   85.001   85.001   85.001  90.00  90.00  90.00 P 21 3 
        if self.space_group_str is None:
            return None
        #print("hi")
        x, cell_dist = self.alignment(segpos, out_cell_spacing=True)
        cell_dist=abs(2*cell_dist)
        return cell_dist, cell_dist, cell_dist, 90, 90, 90, self.space_group_str

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

def Crystal_P213(c3a=None, c3b=None, **kw):
    if c3a is None or c3b is None:
        raise ValueError('must specify ...?') #one or two of c6, c2
    #return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,-1,1,0], [-1,1,1,0], from_seg=c3a, to_seg=c3b, **kw)
    return AxesAngle('Crystal_P213_C3_C3_depth3_1comp', [1,1,1,0], [-1,-1,1,0], from_seg=c3a, to_seg=c3b, space_group_str="P 21 3", **kw)
