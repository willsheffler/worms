"""TODO: Summary
"""

import numpy as np
import numba as nb
import numba.types as nt
from homog import is_homog_xform
from worms import util
from worms.bblock import chain_of_ires
from logging import warning


@nb.jitclass((
    ('x2exit', nt.float64[:, :, :]),
    ('x2orig', nt.float64[:, :, :]),
    ('inout' , nt.int32[:, :]),
    ('inbreaks' , nt.int32[:]),
    ('ires'  , nt.int32[:, :]),
    ('isite' , nt.int32[:, :]),
    ('ichain' , nt.int32[:, :]),
    ('ibb' , nt.int32[:]),
    ('dirn'   , nt.int32[:]),
))  # yapf: disable
class _Vertex:
    """contains data for one topological vertex in the topological graph

    Attributes:
        dirn (TYPE): Description
        ibb (TYPE): Description
        ichain (TYPE): Description
        inout (TYPE): Description
        ires (TYPE): Description
        isite (TYPE): Description
        x2exit (TYPE): Description
        x2orig (TYPE): Description
    """

    def __init__(self, x2exit, x2orig, ires, isite, ichain, ibb, inout,
                 inbreaks, dirn):
        """TODO: Summary

        Args:
            x2exit (TYPE): Description
            x2orig (TYPE): Description
            ires (TYPE): Description
            isite (TYPE): Description
            ichain (TYPE): Description
            ibb (TYPE): Description
            inout (TYPE): Description
            dirn (TYPE): Description

        Deleted Parameters:
            bblock (TYPE): Description
        """
        self.x2exit = x2exit
        self.x2orig = x2orig
        self.ires = ires
        self.isite = isite
        self.ichain = ichain
        self.ibb = ibb
        self.inout = inout
        self.inbreaks = inbreaks
        self.dirn = dirn

    @property
    def len(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return len(self.ires)


def vertex_single(bb, bbid, din, dout, min_seg_len):
    """Summary

    Args:
        bb (TYPE): Description
        bbid (TYPE): Description
        din (TYPE): Description
        dout (TYPE): Description
        min_seg_len (TYPE): Description

    Returns:
        TYPE: Description
    """
    ires0, ires1 = [], []
    isite0, isite1 = [], []
    for i in range(bb.n_connections):
        ires = bb.conn_resids(i)
        if bb.conn_dirn(i) == din:
            ires0.append(ires)
            isite0.append(np.repeat(i, len(ires)))
        if bb.conn_dirn(i) == dout:
            ires1.append(ires)
            isite1.append(np.repeat(i, len(ires)))
    if (din < 2 and not ires0 or dout < 2 and not ires1):
        dirn = 'NC_' [din] + 'NC_' [dout]
        warning('invalid vertex ' + dirn + ' ' + bytes(bb.file).decode())
        return None

    dummy = [np.array([-1], dtype='i4')]
    ires0 = np.concatenate(ires0 or dummy)
    ires1 = np.concatenate(ires1 or dummy)
    isite0 = np.concatenate(isite0 or dummy)
    isite1 = np.concatenate(isite1 or dummy)
    chain0 = chain_of_ires(bb, ires0)
    chain1 = chain_of_ires(bb, ires1)

    if ires0[0] == -1: assert len(ires0) is 1
    else: assert np.all(ires0 >= 0)
    if ires1[0] == -1: assert len(ires1) is 1
    else: assert np.all(ires1 >= 0)

    if ires0[0] is -1: stub0inv = np.eye(4)
    else: stub0inv = np.linalg.inv(bb.stubs[ires0])
    if ires1[0] is -1: stub1 = np.eye(4)
    else: stub1 = bb.stubs[ires1]

    stub0inv, stub1 = np.broadcast_arrays(stub0inv[:, None], stub1)
    ires = np.stack(np.broadcast_arrays(ires0[:, None], ires1), axis=-1)
    isite = np.stack(np.broadcast_arrays(isite0[:, None], isite1), axis=-1)
    chain = np.stack(np.broadcast_arrays(chain0[:, None], chain1), axis=-1)

    x2exit = (stub0inv @ stub1)
    x2orig = stub0inv
    assert is_homog_xform(x2exit)  # this could be slowish
    assert is_homog_xform(x2orig)

    # min chain len, not same site
    not_same_chain = chain[..., 0] != chain[..., 1]
    not_same_site = isite[..., 0] != isite[..., 1]
    seqsep = np.abs(ires[..., 0] - ires[..., 1])

    # remove invalid in/out pairs (+ is or, * is and)
    valid = not_same_site
    valid *= (not_same_chain + (seqsep >= min_seg_len))
    valid = valid.reshape(-1)

    return (
        x2exit.reshape(-1, 4, 4)[valid],
        x2orig.reshape(-1, 4, 4)[valid],
        ires.reshape(-1, 2)[valid].astype('i4'),
        isite.reshape(-1, 2)[valid].astype('i4'),
        chain.reshape(-1, 2)[valid].astype('i4'),
        np.repeat(bbid, sum(valid)).astype('i4'),
    )


def Vertex(bbs, bbids, dirn, min_seg_len=1):
    """Summary

    Args:
        bbs (TYPE): Description
        bbids (TYPE): Description
        dirn (TYPE): Description
        min_seg_len (TYPE): Description

    Returns:
        TYPE: Description
    """
    dirn_map = {'N': 0, 'C': 1, '_': 2}
    din = dirn_map[dirn[0]]
    dout = dirn_map[dirn[1]]

    verts = (vertex_single(bb, bid, din, dout, min_seg_len)
             for bb, bid in zip(bbs, bbids))
    verts = [v for v in verts if v is not None]
    if not verts:
        raise ValueError('no way to make vertex: \'' + dirn + '\'')
    tup = tuple(np.concatenate(_) for _ in zip(*verts))
    assert len({x.shape[0] for x in tup}) == 1
    ibb, ires = tup[5], tup[2]

    inout = np.stack(
        [util.unique_key(ibb, ires[:, 0]),
         util.unique_key(ibb, ires[:, 1])],
        axis=-1).astype('i4')

    inbreaks = util.contig_idx_breaks(inout[:, 0])
    assert inbreaks.dtype == np.int32

    return _Vertex(*tup, inout, inbreaks, np.array([din, dout], dtype='i4'))
