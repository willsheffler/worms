from worms import Vertex
from worms.edge import *
import numba as nb
import numba.types as nt
import numpy as np
import pytest
from worms import vis


@pytest.mark.xfail
def test_Edge(bbdb):
    bbs = bbdb.query('all')
    v = Vertex(bbs, np.arange(len(bbs)), 'NC')
    e = Edge(bbs, bbs)
    print(e)
