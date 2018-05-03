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
    ('dir'   , nt.int32[:]),
))  # yapf: disable
class Vertex:
    """contains data for one topological vertex in the topological graph
    """

    def __init__(self, pdbdat):
        """TODO: Summary

        Args:
            pdbdat (TYPE): Description
        """
        pass


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


@nb.jitclass((
    ('connections', nt.int32[:, :]),
    ('file'       , nt.int8[:]),
    ('components' , nt.int8[:]),
    ('protocol'   , nt.int8[:]),
    ('name'       , nt.int8[:]),
    ('classes'    , nt.int8[:]),
    ('validated'  , nt.boolean),
    ('_type'      , nt.int8[:]),
    ('base'       , nt.int8[:]),
    ('ncac'       , nt.float32[:, :, :]),
    ('chains'     , nt.List(nt.Tuple([nt.int64, nt.int64]))),
    ('ss'         , nt.int8[:]),
    ('stubs'      , nt.float32[:, :, :])
))  # yapf: disable
class BBlock:
    """
    contains data for a single structure building block

    Attributes:
        base (TYPE): Description
        chains (TYPE): Description
        classes (TYPE): Description
        components (TYPE): Description
        connections (TYPE): Description
        file (TYPE): Description
        name (TYPE): Description
        ncac (TYPE): Description
        protocol (TYPE): Description
        ss (TYPE): Description
        stubs (TYPE): Description
        validated (TYPE): Description
    """

    def __init__(self, connections, file, components, protocol, name, classes,
                 validated, _type, base, ncac, chains, ss, stubs):
        """TODO: Summary
        """
        self.connections = connections
        self.file = file
        self.components = components
        self.protocol = protocol
        self.name = name
        self.classes = classes
        self.validated = validated
        self._type = _type
        self.base = base
        self.ncac = ncac
        self.chains = chains
        self.ss = ss
        self.stubs = stubs

    @property
    def n_connections(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.connections)

    def connect_resids(self, i):
        """TODO: Summary

        Args:
            i (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.connections[i, 2:self.connections[i, 1]]

    @property
    def state(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return (self.connections, self.file, self.components, self.protocol,
                self.name, self.classes, self.validated, self._type, self.base,
                self.ncac, self.chains, self.ss, self.stubs)

    # 'connections': [{
    # 'chain': 1,
    # 'residues': ['-150:'],
    # 'direction': 'C'
    # }],


def pdbdat_components(pdbdat):
    """TODO: Summary

    Args:
        pdbdat (TYPE): Description

    Returns:
        TYPE: Description
    """
    return eval(bytes(pdbdat.components))


def pdbdat_str(pdbdat):
    """TODO: Summary

    Args:
        pdbdat (TYPE): Description

    Returns:
        TYPE: Description
    """
    return '\n'.join([
        'jitclass BBlock(',
        '    file=' + str(bytes(pdbdat.file)),
        '    components=' + str(pdbdat_components(pdbdat)),
        '    protocol=' + str(bytes(pdbdat.protocol)),
        '    name=' + str(bytes(pdbdat.name)),
        '    classes=' + str(bytes(pdbdat.classes)),
        '    validated=' + str(pdbdat.validated),
        '    _type=' + str(bytes(pdbdat._type)),
        '    base=' + str(bytes(pdbdat.base)),
        '    ncac=array(shape=' + str(pdbdat.ncac.shape) + ', dtype=' +
        str(pdbdat.ncac.dtype) + ')',
        '    chains=' + str(pdbdat.chains),
        '    ss=array(shape=' + str(pdbdat.ss.shape) + ', dtype=' +
        str(pdbdat.ss.dtype) + ')',
        '    stubs=array(shape=' + str(pdbdat.stubs.shape) + ', dtype=' + str(
            pdbdat.connecions.dtype) + ')',
        '    stubs=array(shape=' + str(pdbdat.connections.shape) + ', dtype=' +
        str(pdbdat.connections.dtype) + ')',
        ')',
    ])
