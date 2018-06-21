import numpy as np


def _validate_bbs_verts(bbs, verts):
    assert len(bbs) == len(verts)
    for bb, vert in zip(bbs, verts):
        assert 0 <= np.min(vert.ibblock)
        assert np.max(vert.ibblock) < len(bb)


class Graph:
    def __init__(self, bbs, verts, edges):
        _validate_bbs_verts(bbs, verts)
        self.bbs = bbs
        self.verts = verts
        self.edges = edges
