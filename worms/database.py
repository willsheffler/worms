import os
import json
from collections import namedtuple
import _pickle as pickle
from worms import *
from concurrent.futures import *
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
    ('connections', nt.int32[:, :]),
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

    @property
    def n_connections(self):
        return len(self.connections)

    def connect_resids(self, i):
        return self.connections[i, 2:self.connections[i, 1]]

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


def _get_connection_residues(entry, chain_bounds):
    chain_bounds[-1][-1]
    r, c, d = entry['residues'], int(entry['chain']), entry['direction']
    if isinstance(r, list):
        try:
            return [int(_) for _ in r]
        except ValueError:
            assert len(r) is 1
            r = r[0]
    if r.startswith('['): return eval(r)
    if r.count(','):
        c2, r = r.split(',')
        assert int(c2) == c
    b, e = r.split(':')
    nres = chain_bounds[c - 1][1] - chain_bounds[c - 1][0]
    b = int(b) if b else 0
    e = int(e) if e else nres
    if e < 0: e += nres
    return list(range(*chain_bounds[c - 1])[b:e])


def make_connections_array(entries, chain_bounds):
    try:
        reslists = [_get_connection_residues(e, chain_bounds) for e in entries]
    except:
        print(entries)
    mx = max(len(x) for x in reslists)
    conn = np.zeros((len(reslists), mx + 2), 'i4') - 1
    for i, rl in enumerate(reslists):
        conn[i, 0] = entries[i]['direction'] == 'C'
        conn[i, 1] = len(rl) + 2
        for j, r in enumerate(rl):
            conn[i, j + 2] = r
    return conn
    # print(chain_bounds)
    # print(repr(conn))


class PDBPile:
    def __init__(self,
                 *,
                 cachedir=None,
                 bakerdb_files=[],
                 load_poses=False,
                 nprocs=1,
                 metaonly=True,
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
        for entry in self.alldb:
            if 'name' not in entry:
                entry['name'] = ''
            entry['file'] = entry['file'].replace(
                '__DATADIR__',
                os.path.dirname(__file__) + '/data')
        self.dictdb = {e['file']: e for e in self.alldb}
        if len(self.alldb) != len(self.dictdb):
            print('!' * 100)
            print('!' * 23,
                  'DIRE WARNING: %6i duplicate pdb files in database' %
                  (len(self.alldb) - len(self.dictdb)), '!' * 23)
            print('!' * 100)
        info('loading %i db entries' % len(self.alldb))
        self.n_new_entries = 0
        self.n_missing_entries = len(self.alldb)
        if not self.metaonly:
            if self.read_new_pdbs:
                assert not os.path.exists(cachedir + '/lock'), (
                    "database is locked! if you're sure no other jobs are editing it, remove "
                    + self.cachedir + "/lock")
                open(cachedir + '/lock', 'w').close()
                assert os.path.exists(cachedir + '/lock')
            self.n_new_entries, self.n_missing_entries = self.load_from_pdbs()
            if self.read_new_pdbs:
                os.remove(cachedir + '/lock')
            if nprocs != 1:
                self.nprocs = 1
                self.load_from_pdbs()

    def __len__(self):
        return len(self.cache)

    def pose(self, pdbfile):
        'load pose from cache, read from file if not in memory'
        if not pdbfile in self.poses:
            if not self.load_cached_pose_into_memory(pdbfile):
                self.poses[pdbfile] = pose_from_file(pdbfile)
        return self.poses[pdbfile]

    def coord(self, pdbfile):
        if not pdbfile in self.coord:
            if not self.load_cached_coords_into_memory(pdbfile):
                self.coord[pdbfile] = pose_from_file(pdbfile)
        return self.coord[pdbfile]

    def query(self, query, *, useclass=True):
        '''
        match name, _type, _class
        if one match, use it
        if _type and _class match, check useclass option
        Het:NNCx/y require exact number or require extra
        '''
        if query.lower() == "all":
            return [db['file'] for db in self.alldb]
        query, subq = query.split(':') if query.count(':') else (query, None)
        if subq is None:
            c_hits = [db['file'] for db in self.alldb if query in db['class']]
            n_hits = [db['file'] for db in self.alldb if query == db['name']]
            t_hits = [db['file'] for db in self.alldb if query == db['type']]
            if not c_hits and not n_hits: return t_hits
            if not c_hits and not t_hits: return n_hits
            if not t_hits and not n_hits: return c_hits
            if not n_hits: return c_hits if useclass else t_hits
            assert False, 'invalid database or query'
        else:
            excon = None
            if subq.endswith('X'): excon = True
            if subq.endswith('Y'): excon = False
            hits = list()
            assert query == 'Het'
            for db in self.alldb:
                if not query in db['class']: continue
                nc = [_ for _ in db['connections'] if _['direction'] == 'C']
                nn = [_ for _ in db['connections'] if _['direction'] == 'N']
                nc, tc = len(nc), subq.count('C')
                nn, tn = len(nn), subq.count('N')
                if nc >= tc and nn >= tn:
                    if nc + nn == tc + tn and excon is not True:
                        hits.append(db['file'])
                    elif nc + nn > tc + tn and excon is not False:
                        hits.append(db['file'])
        return hits

    def load_cached_pose_into_memory(self, pdbfile):
        posefile = self.posefile(pdbfile)
        try:
            with open(posefile, 'rb') as f:
                self.poses[pdbfile] = pickle.load(f)
                return True
        except FileNotFound:
            return False

    def coordfile(self, pdbfile):
        return os.path.join(self.cachedir, 'coord', flatten_path(pdbfile))

    def load_cached_coord_into_memory(self, pdbfile):
        if not isinstance(pdbfile, str):
            success = True
            for f in pdbfile:
                success &= self.load_cached_coord_into_memory(f)
            return success
        coordfile = self.coordfile(pdbfile)
        try:
            with open(coordfile, 'rb') as f:
                self.cache[pdbfile] = pickle.load(f)
                return True
        except FileNotFound:
            return False

    def load_from_pdbs(self):
        if self.nprocs is 1:
            with util.InProcessExecutor() as pool:
                result = [_ for _ in pool.map(self.build_pdb_data, self.alldb)]
        else:
            with ProcessPoolExecutor(max_workers=self.nprocs) as pool:
                result = [_ for _ in pool.map(self.build_pdb_data, self.alldb)]
        return sum(x[0] for x in result), sum(x[1] for x in result)

    def posefile(self, pdbfile):
        return os.path.join(self.cachedir, 'poses', flatten_path(pdbfile))

    def build_pdb_data(self, entry):
        pdbfile = entry['file']
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
            info('PDBPile.build_pdb_data reading %s' % pdbfile)
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
                # connections=np.frombuffer(
                # str(entry['connections']).encode(), dtype='i1'),
                connections=make_connections_array(entry['connections'],
                                                   chains),
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
                info('dumped cache files for', pdbfile)
            self.cache[pdbfile] = pdbdat
            if self.load_poses:
                self.poses[pdbfile] = pose
            return 1, 0
        else:
            print('no cached data for', pdbfile)
            return 0, 1


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
        metaonly=False,
    )
    print('new entries', pp.n_new_entries)
    print('missing entries', pp.n_missing_entries)
    print('total entries', len(pp.cache))
