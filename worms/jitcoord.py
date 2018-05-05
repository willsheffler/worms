"""TODO: Summary
"""
import os
import json
from collections import namedtuple
import _pickle as pickle
from logging import info, error
import itertools as it
import numpy as np
import numba as nb
import numba.types as nt
import pandas as pd

try:
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
except ImportError:
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    error('pyrosetta not available, worms won\'t work')
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')





@nb.jitclass((
    ('x2exit', nt.float32[:, :, :]),
    ('x2orig', nt.float32[:, :, :]),
    ('index' , nt.int32[:, :]),
    ('ires'  , nt.int32[:, :]),
    ('isite' , nt.int32[:, :]),
    ('ichain' , nt.int32[:, :]),
    ('ibb' , nt.int32[:]),
    ('dirn'   , nt.int32[:]),
))  # yapf: disable
class Vertex:
    """contains data for one topological vertex in the topological graph

    Attributes:
        dirn (TYPE): Description
        ibb (TYPE): Description
        ichain (TYPE): Description
        index (TYPE): Description
        ires (TYPE): Description
        isite (TYPE): Description
        x2exit (TYPE): Description
        x2orig (TYPE): Description
    """

    def __init__(self, x2exit, x2orig, ires, isite, ichain, ibb, index, dirn):
        """TODO: Summary

        Args:
            x2exit (TYPE): Description
            x2orig (TYPE): Description
            ires (TYPE): Description
            isite (TYPE): Description
            ichain (TYPE): Description
            ibb (TYPE): Description
            index (TYPE): Description
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
        self.index = index
        self.dirn = dirn

    @property
    def len(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return len(self.ires)


@nb.jitclass((
    ('splices', nt.int32[:, :, :]),
))  # yapf: disable
class Edge:
    """contains junction scores
    """

    def __init__(self, splices):
        """TODO: Summary

        Args:
            splices (TYPE): Description
        """
        pass


def bblock_components(bblock):
    """TODO: Summary

    Args:
        bblock (TYPE): Description

    Returns:
        TYPE: Description
    """
    return eval(bytes(bblock.components))


def bblock_str(bblock):
    """TODO: Summary

    Args:
        bblock (TYPE): Description

    Returns:
        TYPE: Description
    """
    return '\n'.join([
        'jitclass BBlock(',
        '    file=' + str(bytes(bblock.file)),
        '    components=' + str(pdbdat_components(bblock)),
        '    protocol=' + str(bytes(bblock.protocol)),
        '    name=' + str(bytes(bblock.name)),
        '    classes=' + str(bytes(bblock.classes)),
        '    validated=' + str(bblock.validated),
        '    _type=' + str(bytes(bblock._type)),
        '    base=' + str(bytes(bblock.base)),
        '    ncac=array(shape=' + str(bblock.ncac.shape) + ', dtype=' +
        str(bblock.ncac.dtype) + ')',
        '    chains=' + str(bblock.chains),
        '    ss=array(shape=' + str(bblock.ss.shape) + ', dtype=' +
        str(bblock.ss.dtype) + ')',
        '    stubs=array(shape=' + str(bblock.stubs.shape) + ', dtype=' + str(
            bblock.connecions.dtype) + ')',
        '    stubs=array(shape=' + str(bblock.connections.shape) + ', dtype=' +
        str(bblock.connections.dtype) + ')',
        ')',
    ])


@nb.njit
def chain_of_ires(bb, ires):
    """Summary

    Args:
        bb (TYPE): Description
        ires (TYPE): Description

    Returns:
        TYPE: Description
    """
    chain = np.empty_like(ires)
    for i, ir in enumerate(ires):
        for c in range(len(bb.chains)):
            if bb.chains[c, 0] <= ir < bb.chains[c, 1]:
                chain[i] = c
    return chain


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
    ires0 = np.concatenate(ires0)
    ires1 = np.concatenate(ires1)
    isite0 = np.concatenate(isite0)
    isite1 = np.concatenate(isite1)

    chain0 = chain_of_ires(bb, ires0)
    chain1 = chain_of_ires(bb, ires1)

    stb0inv = np.linalg.inv(bb.stubs[ires0])
    stb1 = bb.stubs[ires1]

    stb0inv, stb1 = np.broadcast_arrays(stb0inv, stb1[:, None])
    x2exit = (stb0inv @ stb1)
    x2orig = stb0inv

    ires = np.stack(np.broadcast_arrays(ires0, ires1[:, None]), axis=-1)
    isite = np.stack(np.broadcast_arrays(isite0, isite1[:, None]), axis=-1)
    chain = np.stack(np.broadcast_arrays(chain0, chain1[:, None]), axis=-1)

    # min chain len, not same site
    not_same_chain = chain[..., 0] != chain[..., 1]
    not_same_site = isite[..., 0] != isite[..., 1]
    seqsep = np.abs(ires[..., 0] - ires[..., 1])

    # + is or, * is and
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


def _joint_index(a, b):
    """Summary

    Args:
        a (TYPE): Description
        b (TYPE): Description

    Returns:
        TYPE: Description
    """
    mi = pd.MultiIndex.from_arrays([a, b]).drop_duplicates()
    return mi.get_indexer([a, b])


def vertex(bbs, bbids, dirn, min_seg_len):
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

    tup = tuple(
        np.concatenate(_) for _ in zip(*[
            vertex_single(bb, bid, din, dout, min_seg_len)
            for bb, bid in zip(bbs, bbids)
        ]))
    assert len({x.shape[0] for x in tup}) == 1
    x2exit, x2orig, ires, isite, ichain, ibb = tup

    index = np.stack(
        [_joint_index(ibb, ires[:, 0]),
         _joint_index(ibb, ires[:, 1])],
        axis=-1).astype('i4')

    ####################################

    # print(x2exit.shape)
    # print(x2orig.shape)
    # print(ires.shape)
    # print(isite.shape)
    # print(ichain.shape)
    # print(ibb.shape)
    # print(din, dout)
    return Vertex(*tup, index, np.array([din, dout], dtype='i4'))
