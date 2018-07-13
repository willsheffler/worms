from .base import *
from worms.util import jit
from homog import numba_axis_angle


class Cyclic(WormCriteria):
    def __init__(
            self,
            symmetry=1,
            from_seg=0,
            *,
            tol=1.0,
            origin_seg=None,
            lever=50.0,
            to_seg=-1,
            min_radius=0,
    ):
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
        a = self.symangle

        if self.nfold == 1:
            self.min_sep2 = 0.0
        elif self.nfold == 2:
            self.min_sep2 = 2.0 * min_radius
        else:
            self.min_sep2 = min_radius * np.sin(a) / np.sin((np.pi - a) / 2)
        self.min_sep2 = self.min_sep2**2

    def score(self, segpos, *, verbosity=False, **kw):
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
            carterrsq[np.sum(trans[..., :3]**2, axis=-1) < self.min_sep2] = 9e9

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

    def jit_lossfunc(self):
        tgt_ang = self.symangle
        from_seg = self.from_seg
        to_seg = self.to_seg
        lever = self.lever
        min_sep2 = self.min_sep2

        @jit
        def func(pos, idx, verts):
            x_from = pos[from_seg]
            x_to = pos[to_seg]
            xhat = x_to @ np.linalg.inv(x_from)
            if np.sum(xhat[:3, 3]**2) < min_sep2:
                return 9e9
            axis, angle = numba_axis_angle(xhat)
            rot_err_sq = lever**2 * (angle - tgt_ang)**2
            cart_err_sq = (np.sum(xhat[:, 3] * axis))**2
            return np.sqrt(rot_err_sq + cart_err_sq)

        return func
