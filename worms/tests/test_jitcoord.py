from worms.jitcoord import *
import numba as nb
import numba.types as nt
import numpy as np


def test_Vertex(pdbpile):
    bbs = pdbpile.query('all')
    v = vertex(bbs, np.arange(len(bbs)), 'NC', 5)
    assert v.len == 55
    assert v.x2exit.shape == (55, 4, 4)
    assert v.x2orig.shape == (55, 4, 4)
    assert v.index.shape == (55, 2)
    assert v.ires.shape == (55, 2)
    assert v.isite.shape == (55, 2)
    assert v.ichain.shape == (55, 2)
    assert v.ibb.shape == (55, )
    assert v.dirn.shape == (2, )
