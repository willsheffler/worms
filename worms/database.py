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
    ('name', nt.int8[:]),
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
            name,
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
        self.name = name
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
            self.name,
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
            pdbdat.stubs.dtype) + ')',
        ')',
    ])


def flatten_path(pdbfile):
    return pdbfile.replace(os.sep, '__') + '.pickle'


class PDBPile:
    def __init__(self,
                 *,
                 cachedir=None,
                 bakerdb_files=[],
                 load_poses=False,
                 nprocs=1,
                 metaonly=False,
                 read_new_pdbs=False):
        if cachedir is None:
            if 'HOME' in os.environ:
                cachedir = os.environ['HOME'] + os.sep + '.worms/cache'
            else:
                cachedir = '.worms/cache'
        self.cachedir = str(cachedir)
        self.load_poses = load_poses
        os.makedirs(self.cachedir + '/poses', exist_ok=True)
        os.makedirs(self.cachedir + '/coord', exist_ok=True)
        self.cache, self.poses = dict(), dict()
        self.nprocs = nprocs
        self.metaonly = metaonly
        self.read_new_pdbs = read_new_pdbs
        self.alldb = []
        for dbfile in bakerdb_files:
            with open(dbfile) as f:
                self.alldb.extend(json.load(f))
        self.dictdb = {e['file']: e for e in self.alldb}
        if len(self.alldb) != len(self.dictdb):
            print('!' * 100)
            print('!' * 23,
                  'DIRE WARNING: %6i duplicate pdb files in database' %
                  (len(self.alldb) - len(self.dictdb)), '!' * 23)
            print('!' * 100)
        print('loading %i db entries' % len(self.alldb))
        self.n_new_entries = 0
        self.n_missing_entries = len(self.alldb)
        if not self.metaonly:
            if self.read_new_pdbs:
                assert not os.path.exists(cachedir + '/lock'), (
                    "database is locked! if you're sure no other jobs are editing it, remove "
                    + self.cachedir + "/lock")
                open(cachedir + '/lock', 'w').close()
                assert os.path.exists(cachedir + '/lock')
            self.n_new_entries, self.n_missing_entries = self.load()
            if self.read_new_pdbs:
                os.remove(cachedir + '/lock')
            if nprocs != 1:
                self.nprocs = 1
                self.load()

    def __len__(self):
        return len(self.cache)

    def find_by_class(self, _class):
        c = _class
        subc = None
        if _class.count(':'):
            c, subc = _class.split(':')
        if subc is None:
            hits = [db['file'] for db in self.alldb if c in db['class']]
        else:
            hits = list()
            assert c == 'Het'
            for db in self.alldb:
                if not c in db['class']: continue
                nc = [x for x in db['connections'] if x['direction'] == 'C']
                nn = [x for x in db['connections'] if x['direction'] == 'N']
                if subc.count('C') <= len(nc) and subc.count('N') <= len(nn):
                    hits.append(db['file'])
        return hits

    def query(self, s):
        '''
        match name, _type, _class
        if one match, use it
        if _type and _class match, check useclass option
        Het:NNCx/y require exact number or require extra
        '''
        pass

    def classes(self):
        x = set()
        for db in self.alldb:
            x.update(db['class'])
        return list(x)

    def load(self):
        if self.nprocs is 1:
            with util.InProcessExecutor() as pool:
                result = [_ for _ in pool.map(self.add_entry, self.alldb)]
        else:
            with ProcessPoolExecutor(max_workers=self.nprocs) as pool:
                result = [_ for _ in pool.map(self.add_entry, self.alldb)]
        return sum(x[0] for x in result), sum(x[1] for x in result)

    def posefile(self, pdbfile):
        return os.path.join(self.cachedir, 'poses', flatten_path(pdbfile))

    def load_cached_pose_into_memory(self, pdbfile):
        posefile = self.posefile(pdbfile)
        try:
            with open(posefile, 'rb') as f:
                self.poses[pdbfile] = pickle.load(f)
                return True
        except FileNotFound:
            return False

    def add_entry(self, entry):
        pdbfile = entry['file']
        if 'name' not in entry: entry['name'] = ''
        cachefile = os.path.join(self.cachedir, 'coord', flatten_path(pdbfile))
        posefile = self.posefile(pdbfile)
        if os.path.exists(cachefile):
            with open(cachefile, 'rb') as f:
                pdbdat = PDBDat(*pickle.load(f))
            if self.load_poses:
                assert self.load_cached_pose_into_memory(pdbfile)
            self.cache[pdbfile] = pdbdat
            if self.load_poses:
                self.poses[pdbfile] = pose
            return 0, 0
        elif self.read_new_pdbs:
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
                name=np.frombuffer(entry['name'].encode(), dtype='i1'),
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
                    assert bytes(tmp.name) == bytes(pdbdat.name)
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
                print('dumped cache files for', pdbfile)
                sys.stdout.flush()
            self.cache[pdbfile] = pdbdat
            if self.load_poses:
                self.poses[pdbfile] = pose
            return 1, 0
        else:
            print('no cached data for', pdbfile)
            return 0, 1

    # def compress_poses(self):
    # for k, pose in self.poses.items():
    # if not isinstance(pose, bytes):
    # self.poses[k] = pickle.dumps(pose)

    def pose(self, pdbfile):
        'load pose from cache, read from file if not in memory'
        if not pdbfile in self.poses:
            if not self.load_cached_pose_into_memory(pdbfile):
                self.poses[pdbfile] = pose_from_file(pdbfile)
        return self.poses[pdbfile]


if __name__ == '__main__':
    import argparse
    import pyrosetta
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--add_database_files_to_cache',
        type=str,
        nargs='+',
        dest='database_files')
    parser.add_argument('--nprocs', type=int, dest='nprocs', default=1)
    parser.add_argument(
        '--read_new_pdbs', type=bool, dest='read_new_pdbs', default=False)
    args = parser.parse_args()
    pyrosetta.init('-mute all -ignore_unrecognized_res')
    pp = PDBPile(
        bakerdb_files=args.database_files,
        nprocs=args.nprocs,
        read_new_pdbs=args.read_new_pdbs,
    )
    print('new entries', pp.n_new_entries)
    print('missing entries', pp.n_missing_entries)
    print('total entries', len(pp.cache))
