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
            kw: passthru args        """
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
            kw: passthru args
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
            kw: passthru args
        Returns:
            TYPE: Description
        """
        return np.zeros(segpos[-1].shape[:-2])

    def alignment(self, segpos, **kw):
        """TODO: Summary

        Args:
            segpos (TYPE): Description
            kw: passthru args
        Returns:
            TYPE: Description
        """
        r = np.empty_like(segpos[-1])
        r[..., :, :] = np.eye(4)
        return r
