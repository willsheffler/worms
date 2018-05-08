from .base import *


class Cyclic(WormCriteria):
    """TODO: Summary

    Attributes:
        from_seg (TYPE): Description
        is_cyclic (bool): Description
        last_body_same_as (TYPE): Description
        lever (TYPE): Description
        nfold (TYPE): Description
        origin_seg (TYPE): Description
        rot_tol (TYPE): Description
        sym_axes (TYPE): Description
        symangle (TYPE): Description
        symmetry (TYPE): Description
        symname (TYPE): Description
        to_seg (TYPE): Description
        tol (TYPE): Description
    """

    def __init__(self,
                 symmetry=1,
                 from_seg=0,
                 *,
                 tol=1.0,
                 origin_seg=None,
                 lever=50.0,
                 to_seg=-1):
        """TODO: Summary

        Args:
            symmetry (int, optional): Description
            from_seg (int, optional): Description
            tol (float, optional): Description
            origin_seg (None, optional): Description
            lever (float, optional): Description
            to_seg (TYPE, optional): Description

        Raises:
            ValueError: Description
        """
        if from_seg == to_seg:
            raise ValueError('from_seg should not be same as to_seg')
        if from_seg == origin_seg:
            raise ValueError('from_seg should not be same as origin_seg')
        if to_seg == origin_seg:
            raise ValueError('to_seg should not be same as origin_seg')
        if isinstance(symmetry, int): symmetry = 'C' + str(symmetry)
        self.symmetry = symmetry
        self.tol = tol
        self.from_seg = from_seg
        self.origin_seg = origin_seg
        self.lever = lever
        self.to_seg = to_seg
        self.rot_tol = tol / lever
        # self.relweight = relweight if abs(relweight) > 0.001 else None
        if self.symmetry[0] in 'cC':
            self.nfold = int(self.symmetry[1:])
            if self.nfold <= 0:
                raise ValueError('invalid symmetry: ' + symmetry)
            self.symangle = np.pi * 2.0 / self.nfold
        else:
            raise ValueError('can only do Cx symmetry for now')
        if self.tol <= 0: raise ValueError('tol should be > 0')
        self.last_body_same_as = self.from_seg
        self.is_cyclic = True
        self.symname = None
        if self.nfold > 1:
            self.symname = 'C' + str(self.nfold)
        self.sym_axes = [(self.nfold, Uz, [0, 0, 0, 1])]

    def score(self, segpos, *, verbosity=False, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            verbosity (bool, optional): Description
            kw: passthru args
        Returns:
            TYPE: Description
        """
        x_from = segpos[self.from_seg]
        x_to = segpos[self.to_seg]
        xhat = x_to @ inv(x_from)
        trans = xhat[..., :, 3]
        if self.nfold is 1:
            angle = hm.angle_of(xhat)
            carterrsq = np.sum(trans[..., :3]**2, axis=-1)
            roterrsq = angle**2
        else:
            if self.origin_seg is not None:
                tgtaxis = segpos[self.origin_seg] @ [0, 0, 1, 0]
                tgtcen = segpos[self.origin_seg] @ [0, 0, 0, 1]
                axis, angle, cen = hm.axis_ang_cen_of(xhat)
                carterrsq = hm.hnorm2(cen - tgtcen)
                roterrsq = (1 - np.abs(hm.hdot(axis, tgtaxis))) * np.pi
            else:  # much cheaper if cen not needed
                axis, angle = hm.axis_angle_of(xhat)
                carterrsq = roterrsq = 0
            carterrsq = carterrsq + hm.hdot(trans, axis)**2
            roterrsq = roterrsq + (angle - self.symangle)**2
            # if self.relweight is not None:
            #     # penalize 'relative' error
            #     distsq = np.sum(trans[..., :3]**2, axis=-1)
            #     relerrsq = carterrsq / distsq
            #     relerrsq[np.isnan(relerrsq)] = 9e9
            #     # too much of a hack??
            #     carterrsq += self.relweight * relerrsq
            if verbosity > 0:
                print('axis', axis[0])
                print('trans', trans[0])
                print('dot trans', hm.hdot(trans, axis)[0])
                print('angle', angle[0] * 180 / np.pi)

        return np.sqrt(carterrsq / self.tol**2 + roterrsq / self.rot_tol**2)

    def alignment(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            kw: passthru args

        Returns:
            TYPE: Description
        """
        if self.origin_seg is not None:
            return inv(segpos[self.origin_seg])
        x_from = segpos[self.from_seg]
        x_to = segpos[self.to_seg]
        xhat = x_to @ inv(x_from)
        axis, ang, cen = hm.axis_ang_cen_of(xhat)
        # print('aln', axis)
        # print('aln', ang * 180 / np.pi)
        # print('aln', cen)
        # print('aln', xhat[..., :, 3])
        dotz = hm.hdot(axis, Uz)[..., None]
        tgtaxis = np.where(dotz > 0, [0, 0, 1, 0], [0, 0, -1, 0])
        align = hm.hrot((axis + tgtaxis) / 2, np.pi, cen)
        align[..., :3, 3] -= cen[..., :3]
        return align
