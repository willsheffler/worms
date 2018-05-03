"""TODO: Summary

Attributes:
    Ux (TYPE): Description
    Uy (TYPE): Description
    Uz (TYPE): Description
"""
import abc
import numpy as np
import homog as hm
from numpy.linalg import inv

Ux = np.array([1, 0, 0, 0])
Uy = np.array([0, 1, 0, 0])
Uz = np.array([0, 0, 1, 0])


class WormCriteria(abc.ABC):
    """TODO: Summary

    Attributes:
        allowed_attributes (TYPE): Description
    """

    @abc.abstractmethod
    def score(self, **kw):
        """TODO: Summary

        Args:
            **kw: Description
        """
        pass

    allowed_attributes = (
        'last_body_same_as',
        'symname',
        'is_cyclic',
        'alignment',
        'from_seg',
        'to_seg',
        'origin_seg',
        'symfile_modifiers',
    )


class CriteriaList(WormCriteria):
    """TODO: Summary

    Attributes:
        children (TYPE): Description
    """

    def __init__(self, children):
        """TODO: Summary

        Args:
            children (TYPE): Description
        """
        if isinstance(children, WormCriteria):
            children = [children]
        self.children = children

    def score(self, **kw):
        """TODO: Summary

        Args:
            **kw: Description

        Returns:
            TYPE: Description
        """
        return sum(c.score(**kw) for c in self.children)

    def __getattr__(self, name):
        """TODO: Summary

        Args:
            name (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            AttributeError: Description
        """
        if name not in WormCriteria.allowed_attributes:
            raise AttributeError('CriteriaList has no attribute: ' + name)
        r = [getattr(c, name) for c in self.children if hasattr(c, name)]
        r = [x for x in r if x is not None]
        assert len(r) < 2
        return r[0] if len(r) else None

    def __getitem__(self, index):
        """TODO: Summary

        Args:
            index (TYPE): Description

        Returns:
            TYPE: Description
        """
        assert isinstance(index, int)
        return self.children[index]

    def __len__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.children)

    def __iter__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return iter(self.children)


class NullCriteria(WormCriteria):
    """TODO: Summary

    Attributes:
        from_seg (TYPE): Description
        to_seg (TYPE): Description
    """

    def __init__(self, from_seg=0, to_seg=-1, origin_seg=None):
        """TODO: Summary

        Args:
            from_seg (int, optional): Description
            to_seg (TYPE, optional): Description
            origin_seg (None, optional): Description
        """
        self.from_seg = from_seg
        self.to_seg = to_seg

    def score(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            **kw: Description

        Returns:
            TYPE: Description
        """
        return np.zeros(segpos[-1].shape[:-2])

    def alignment(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            **kw: Description

        Returns:
            TYPE: Description
        """
        r = np.empty_like(segpos[-1])
        r[..., :, :] = np.eye(4)
        return r


class AxesIntersect(WormCriteria):
    """TODO: Summary

    Attributes:
        angle (TYPE): Description
        distinct_axes (TYPE): Description
        from_seg (TYPE): Description
        lever (TYPE): Description
        rot_tol (TYPE): Description
        sym_axes (TYPE): Description
        symname (TYPE): Description
        tgtaxis1 (TYPE): Description
        tgtaxis2 (TYPE): Description
        to_seg (TYPE): Description
        tol (TYPE): Description
    """

    def __init__(self,
                 symname,
                 tgtaxis1,
                 tgtaxis2,
                 from_seg,
                 *,
                 tol=1.0,
                 lever=50,
                 to_seg=-1,
                 distinct_axes=False):
        """TODO: Summary

        Args:
            symname (TYPE): Description
            tgtaxis1 (TYPE): Description
            tgtaxis2 (TYPE): Description
            from_seg (TYPE): Description
            tol (float, optional): Description
            lever (int, optional): Description
            to_seg (TYPE, optional): Description
            distinct_axes (bool, optional): Description

        Raises:
            ValueError: Description
        """
        if from_seg == to_seg:
            raise ValueError('from_seg should not be same as to_seg')
        self.symname = symname
        self.from_seg = from_seg
        if len(tgtaxis1) == 2: tgtaxis1 += [0, 0, 0, 1],
        if len(tgtaxis2) == 2: tgtaxis2 += [0, 0, 0, 1],
        self.tgtaxis1 = (tgtaxis1[0], hm.hnormalized(tgtaxis1[1]),
                         hm.hpoint(tgtaxis1[2]))
        self.tgtaxis2 = (tgtaxis2[0], hm.hnormalized(tgtaxis2[1]),
                         hm.hpoint(tgtaxis2[2]))
        assert 3 == len(self.tgtaxis1)
        assert 3 == len(self.tgtaxis2)
        self.angle = hm.angle(tgtaxis1[1], tgtaxis2[1])
        self.tol = tol
        self.lever = lever
        self.to_seg = to_seg
        self.rot_tol = tol / lever
        self.distinct_axes = distinct_axes  # -z not same as z (for T33)
        self.sym_axes = [self.tgtaxis1, self.tgtaxis2]

    def score(self, segpos, verbosity=False, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            verbosity (bool, optional): Description
            **kw: Description

        Returns:
            TYPE: Description
        """
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
        roterr2 = (ang - self.angle)**2
        return np.sqrt(roterr2 / self.rot_tol**2 + (dist / self.tol)**2)

    def alignment(self, segpos, debug=0, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            debug (int, optional): Description
            **kw: Description

        Returns:
            TYPE: Description

        Raises:
            AssertionError: Description
        """
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
            print('angs', hm.angle_degrees(ax1, ax2),
                  hm.angle_degrees(self.tgtaxis1[1], self.tgtaxis2[1]))
            print('ax1', ax1)
            print('ax2', ax2)
            print('xax1', x @ ax1)
            print('tax1', self.tgtaxis1[1])
            print('xax2', x @ ax2)
            print('tax2', self.tgtaxis2[1])
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


def D2(c2=0, c2b=-1, **kw):
    """TODO: Summary

    Args:
        c2 (int, optional): Description
        c2b (TYPE, optional): Description
        **kw: Description

    Returns:
        TYPE: Description
    """
    return AxesIntersect('D2', (2, Uz), (2, Ux), c2, to_seg=c2b, **kw)


def D3(c3=0, c2=-1, **kw):
    """TODO: Summary

    Args:
        c3 (int, optional): Description
        c2 (TYPE, optional): Description
        **kw: Description

    Returns:
        TYPE: Description
    """
    return AxesIntersect('D3', (3, Uz), (2, Ux), c3, to_seg=c2, **kw)


def D4(c4=0, c2=-1, **kw):
    """TODO: Summary

    Args:
        c4 (int, optional): Description
        c2 (TYPE, optional): Description
        **kw: Description

    Returns:
        TYPE: Description
    """
    return AxesIntersect('D4', (4, Uz), (2, Ux), c4, to_seg=c2, **kw)


def D5(c5=0, c2=-1, **kw):
    """TODO: Summary

    Args:
        c5 (int, optional): Description
        c2 (TYPE, optional): Description
        **kw: Description

    Returns:
        TYPE: Description
    """
    return AxesIntersect('D5', (5, Uz), (2, Ux), c5, to_seg=c2, **kw)


def D6(c6=0, c2=-1, **kw):
    """TODO: Summary

    Args:
        c6 (int, optional): Description
        c2 (TYPE, optional): Description
        **kw: Description

    Returns:
        TYPE: Description
    """
    return AxesIntersect('D6', (6, Uz), (2, Ux), c6, to_seg=c2, **kw)


def Tetrahedral(c3=None, c2=None, c3b=None, **kw):
    """TODO: Summary

    Args:
        c3 (None, optional): Description
        c2 (None, optional): Description
        c3b (None, optional): Description
        **kw: Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if 1 is not (c3b is None) + (c3 is None) + (c2 is None):
        raise ValueError('must specify exactly two of c3, c2, c3b')
    if c2 is None: from_seg, to_seg, nf1, nf2, ex = c3b, c3, 7, 3, 2
    if c3 is None: from_seg, to_seg, nf1, nf2, ex = c3b, c2, 7, 2, 3
    if c3b is None: from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 7
    return AxesIntersect(
        'T',
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(max(3, nf1), hm.sym.tetrahedral_axes[nf1]),
        tgtaxis2=(max(3, nf2), hm.sym.tetrahedral_axes[nf2]),
        distinct_axes=(nf1 == 7),
        **kw)


def Octahedral(c4=None, c3=None, c2=None, **kw):
    """TODO: Summary

    Args:
        c4 (None, optional): Description
        c3 (None, optional): Description
        c2 (None, optional): Description
        **kw: Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if 1 is not (c4 is None) + (c3 is None) + (c2 is None):
        raise ValueError('must specify exactly two of c4, c3, c2')
    if c2 is None: from_seg, to_seg, nf1, nf2, ex = c4, c3, 4, 3, 2
    if c3 is None: from_seg, to_seg, nf1, nf2, ex = c4, c2, 4, 2, 3
    if c4 is None: from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 4
    return AxesIntersect(
        'O',
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(nf1, hm.sym.octahedral_axes[nf1]),
        tgtaxis2=(nf2, hm.sym.octahedral_axes[nf2]),
        **kw)


def Icosahedral(c5=None, c3=None, c2=None, **kw):
    """TODO: Summary

    Args:
        c5 (None, optional): Description
        c3 (None, optional): Description
        c2 (None, optional): Description
        **kw: Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if 1 is not (c5 is None) + (c3 is None) + (c2 is None):
        raise ValueError('must specify exactly two of c5, c3, c2')
    if c2 is None: from_seg, to_seg, nf1, nf2, ex = c5, c3, 5, 3, 2
    if c3 is None: from_seg, to_seg, nf1, nf2, ex = c5, c2, 5, 2, 3
    if c5 is None: from_seg, to_seg, nf1, nf2, ex = c3, c2, 3, 2, 5
    return AxesIntersect(
        'I',
        from_seg=from_seg,
        to_seg=to_seg,
        tgtaxis1=(nf1, hm.sym.icosahedral_axes[nf1]),
        tgtaxis2=(nf2, hm.sym.icosahedral_axes[nf2]),
        **kw)


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
            **kw: Description

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

    def alignment(self, segpos, **kwargs):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            **kwargs: Description

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
