from worms.jitcoord import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis


def test_Vertex(bbdb):
    bbs = bbdb.query('all')
    v = Vertex(bbs, np.arange(len(bbs)), 'NC', 5)
    assert v.len == 55
    assert v.x2exit.shape == (55, 4, 4)
    assert v.x2orig.shape == (55, 4, 4)
    assert v.index.shape == (55, 2)
    assert v.ires.shape == (55, 2)
    assert v.isite.shape == (55, 2)
    assert v.ichain.shape == (55, 2)
    assert v.ibb.shape == (55, )
    assert v.dirn.shape == (2, )


@pytest.mark.xfail
def test_Edge(bbdb):

    assert 0
