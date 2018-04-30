import os
import json
from collections import namedtuple
import _pickle as pickle
from worms import *
from concurrent.futures import *
from logging import info, error
import itertools as it
import numpy as np
import numba
import numba.types as nt

try:
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
except ImportError:
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    error('pyrosetta not available, worms won\'t work')
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


@numba.jitclass((
    ('connections', nt.int8[:]),
    ('file', nt.int8[:]),
    ('components', nt.int8[:]),
    ('protocol', nt.int8[:]),
    ('classes', nt.int8[:]),
    ('validated', nt.boolean),
    ('_type', nt.int8[:]),
    ('base', nt.int8[:]),
    ('ncac', nt.float32[:, :, :]),
    ('chains', nt.List(nt.Tuple([nt.int64, nt.int64]))),
    ('ss', nt.int8[:]),
    ('stubs', nt.float32[:, :, :]),
))
class PDBDat:
    def __init__(
            self,
            connections,
            file,
            components,
            protocol,
            classes,
            validated,
            _type,
            base,
            ncac,
            chains,
            ss,
            stubs,
    ):
        self.connections = connections
        self.file = file
        self.components = components
        self.protocol = protocol
        self.classes = classes
        self.validated = validated
        self._type = _type
        self.base = base
        self.ncac = ncac
        self.chains = chains
        self.ss = ss
        self.stubs = stubs

    def getstate(self):
        return (
            self.connections,
            self.file,
            self.components,
            self.protocol,
            self.classes,
            self.validated,
            self._type,
            self.base,
            self.ncac,
            self.chains,
            self.ss,
            self.stubs,
        )

    # 'connections': [{
    # 'chain': 1,
    # 'residues': ['-150:'],
    # 'direction': 'C'
    # }],


def pdbdat_connections(pdbdat):
    return eval(bytes(pdbdat.connections))


def pdbdat_components(pdbdat):
    return eval(bytes(pdbdat.components))


def pdbdat_str(pdbdat):
    return '\n'.join([
        'jitclass PDBDat(',
        '    connections=' + str(pdbdat_connections(pdbdat)),
        '    file=' + str(bytes(pdbdat.file)),
        '    components=' + str(pdbdat_components(pdbdat)),
        '    protocol=' + str(bytes(pdbdat.protocol)),
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
            pdbdat.stubs.dtype) + ')',
        ')',
    ])


def flatten_path(pdbfile):
    return pdbfile.replace(os.sep, '__') + '.pickle'


class PDBPile:
    def __init__(self,
                 *,
                 cachedir=os.environ['HOME'] + os.sep + '.worms/cache',
                 bakerdb_files=[],
                 load_poses=False,
                 exe=ThreadPoolExecutor):
        self.cachedir = str(cachedir)
        self.load_poses = load_poses
        os.makedirs(self.cachedir + '/poses', exist_ok=True)
        os.makedirs(self.cachedir + '/coord', exist_ok=True)
        self.cache, self.poses = dict(), dict()
        self.exe = exe
        for f in bakerdb_files:
            self.add_bakerdb_file(f)

    def add_bakerdb_file(self, json_db_file):
        with open(json_db_file) as f:
            db = json.load(f)
            info('PDBPile.add_bakerdb_file %s %i' % (json_db_file, len(db)))
            with self.exe() as pool:
                [_ for _ in pool.map(self.add_entry, db)]

    def posefile(self, pdbfile):
        return os.path.join(self.cachedir, 'poses', flatten_path(pdbfile))

    def load_cached_pose_into_memory(self, pdbfile):
        posefile = self.posefile(pdbfile)
        with open(posefile, 'rb') as f:
            self.poses[pdbfile] = pickle.load(f)

    def add_entry(self, entry):
        pdbfile = entry['file']
        cachefile = os.path.join(self.cachedir, 'coord', flatten_path(pdbfile))
        posefile = self.posefile(pdbfile)
        if os.path.exists(cachefile):
            with open(cachefile, 'rb') as f:
                pdbdat = PDBDat(*pickle.load(f))
            if self.load_poses:
                self.load_cached_pose_into_memory(pdbfile)
        else:
            info('PDBPile.add_entry reading %s' % pdbfile)
            if os.path.exists(posefile):
                with open(posefile, 'rb') as f:
                    pose = pickle.load(f)
            else:
                pose = pose_from_file(pdbfile)
            chains = util.get_chain_bounds(pose)
            ss = np.frombuffer(
                str(Dssp(pose).get_dssp_secstruct()).encode(), dtype='i1')
            stubs, ncac = util.get_bb_stubs(pose)
            assert len(pose) == len(ncac)
            assert len(pose) == len(stubs)
            assert len(pose) == len(ss)
            pdbdat = PDBDat(
                connections=np.frombuffer(
                    str(entry['connections']).encode(), dtype='i1'),
                file=np.frombuffer(entry['file'].encode(), dtype='i1'),
                components=np.frombuffer(
                    str(entry['components']).encode(), dtype='i1'),
                protocol=np.frombuffer(entry['protocol'].encode(), dtype='i1'),
                classes=np.frombuffer(','.join(entry['class']).encode(), 'i1'),
                validated=entry['validated'],
                _type=np.frombuffer(entry['type'].encode(), dtype='i1'),
                base=np.frombuffer(entry['base'].encode(), dtype='i1'),
                ncac=ncac.astype('f4'),
                chains=chains,
                ss=ss,
                stubs=stubs.astype('f4'),
            )
            assert pdbdat.chains == chains
            with open(cachefile, 'wb') as f:
                pickle.dump(pdbdat.getstate(), f)
            sanity_check = True
            if sanity_check:
                with open(cachefile, 'rb') as f:
                    tup = tuple(pickle.load(f))
                    tmp = PDBDat(*tup)
                    # print('-' * 40, 'orig', '-' * 40)
                    # print(pdbdat_str(pdbdat))
                    # print('-' * 40, 'reread', '-' * 40)
                    # print(pdbdat_str(tmp))
                    # print('-' * 100)
                    assert bytes(tmp.connections) == bytes(pdbdat.connections)
                    assert bytes(tmp.file) == bytes(pdbdat.file)
                    assert bytes(tmp.components) == bytes(pdbdat.components)
                    assert bytes(tmp.protocol) == bytes(pdbdat.protocol)
                    assert bytes(tmp.classes) == bytes(pdbdat.classes)
                    assert tmp.validated == pdbdat.validated
                    assert bytes(tmp._type) == bytes(pdbdat._type)
                    assert bytes(tmp.base) == bytes(pdbdat.base)
                    assert np.allclose(tmp.ncac, pdbdat.ncac)
                    assert pdbdat.chains == tmp.chains
                    assert np.allclose(tmp.ss, pdbdat.ss)
                    assert np.allclose(tmp.stubs, pdbdat.stubs)
            with open(posefile, 'wb') as f:
                pickle.dump(pose, f)
        self.cache[pdbfile] = pdbdat
        if self.load_poses:
            self.poses[pdbfile] = pose

    # def compress_poses(self):
    # for k, pose in self.poses.items():
    # if not isinstance(pose, bytes):
    # self.poses[k] = pickle.dumps(pose)

    def pose(self, pdbfile):
        'load pose from cache, read from file if not in memory'
        if not pdbfile in self.poses:
            self.load_cached_pose_into_memory(pdbfile)
        pose = self.poses[pdbfile]
        # if isinstance(pose, bytes):
        # pose = pickle.loads(pose)
        return pose


if __name__ == '__main__':
    import argparse
    import pyrosetta
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--database_files', type=str, nargs='+', dest='database_files')
    args = parser.parse_args()
    pyrosetta.init('-mute all -ignore_unrecognized_res')
    pp = PDBPile(bakerdb_files=args.database_files, exe=ProcessPoolExecutor)
    print(len(pp.cache))
