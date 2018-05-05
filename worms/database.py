"""TODO: Summary
"""
import os
import json
import _pickle as pickle
from concurrent.futures import *
import itertools as it
import logging
from logging import info, error
from random import shuffle

import numpy as np
from tqdm import tqdm

from worms import util
from worms import BBlock
from worms.BBlock import _BBlock

logging.basicConfig(level=logging.INFO)

try:
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
except ImportError:
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    error('pyrosetta not available, worms won\'t work')
    error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


def flatten_path(pdbfile):
    """TODO: Summary

    Args:
        pdbfile (TYPE): Description

    Returns:
        TYPE: Description
    """
    return pdbfile.replace(os.sep, '__') + '.pickle'


class PDBPile:
    """TODO: Summary

    Attributes:
        alldb (list): Description
        cachedir (TYPE): Description
        dictdb (TYPE): Description
        load_poses (TYPE): Description
        metaonly (TYPE): Description
        n_missing_entries (TYPE): Description
        n_new_entries (int): Description
        nprocs (int): Description
        read_new_pdbs (TYPE): Description
    """

    def __init__(
            self,
            cachedir=None,
            bakerdb_files=[],
            load_poses=False,
            nprocs=1,
            metaonly=True,
            read_new_pdbs=False,
    ):
        """TODO: Summary

        Args:
            cachedir (None, optional): Description
            bakerdb_files (list, optional): Description
            load_poses (bool, optional): Description
            nprocs (int, optional): Description
            metaonly (bool, optional): Description
            read_new_pdbs (bool, optional): Description
        """
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
        for i, k in enumerate(sorted(self.dictdb)):
            self.alldb[i] = self.dictdb[k]

    def __getitem__(self, i):
        if isinstance(i, str):
            return self.cache[i]
        else:
            return self.cache.values()[i]

    def __len__(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return len(self.cache)

    def pose(self, pdbfile):
        """load pose from cache, read from file if not in memory

        Args:
            pdbfile (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not pdbfile in self.poses:
            if not self.load_cached_pose_into_memory(pdbfile):
                self.poses[pdbfile] = pose_from_file(pdbfile)
        return self.poses[pdbfile]

    def coord(self, pdbfile):
        """TODO: Summary

        Args:
            pdbfile (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not pdbfile in self.coord:
            if not self.load_cached_coords_into_memory(pdbfile):
                self.coord[pdbfile] = pose_from_file(pdbfile)
        return self.coord[pdbfile]

    def query(self, query, *, useclass=True):
        """
        match name, _type, _class
        if one match, use it
        if _type and _class match, check useclass option
        Het:NNCx/y require exact number or require extra

        Args:
            query (TYPE): Description
            useclass (bool, optional): Description

        Returns:
            TYPE: Description
        """
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
        """TODO: Summary

        Args:
            pdbfile (TYPE): Description

        Returns:
            TYPE: Description
        """
        posefile = self.posefile(pdbfile)
        try:
            with open(posefile, 'rb') as f:
                self.poses[pdbfile] = pickle.load(f)
                return True
        except FileNotFoundError:
            return False

    def coordfile(self, pdbfile):
        """TODO: Summary"""
        return os.path.join(self.cachedir, 'coord', flatten_path(pdbfile))

    def load_cached_coord_into_memory(self, pdbfile):
        """TODO: Summary

        Args:
            pdbfile (TYPE): Description

        Returns:
            TYPE: Description
        """
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

    def posefile(self, pdbfile):
        """TODO: Summary"""
        return os.path.join(self.cachedir, 'poses', flatten_path(pdbfile))

    def load_from_pdbs(self):
        """Summary

       Returns:
           TYPE: Description
       """
        shuffle(self.alldb)
        if self.nprocs is 1:
            with util.InProcessExecutor() as exe:
                result = self.load_from_pdbs_inner(exe)
        else:
            with ProcessPoolExecutor(max_workers=self.nprocs) as exe:
                result = self.load_from_pdbs_inner(exe)
        new = [_[0] for _ in result if _[0]]
        missing = [_[1] for _ in result if _[1]]
        for miss in missing:
            self.alldb.remove(self.dictdb[miss])
            del self.dictdb[miss]
        return len(new), len(missing)

    def load_from_pdbs_inner(self, exe):
        """Summary

       Args:
           exe (TYPE): Description

       Returns:
           TYPE: Description
       """
        # return exe.map(self.build_pdb_data, self.alldb)
        shuffle(self.alldb)
        r = []
        print('load_from_pdbs', len(self.alldb))
        kwargs = {
            'total': len(self.alldb),
            'unit': 'pdbs',
            # 'unit_scale': True,
            'leave': True
        }
        futures = [exe.submit(self.build_pdb_data, e) for e in self.alldb]
        for f in tqdm(as_completed(futures), **kwargs):
            r.append(f.result())
        return r

    def build_pdb_data(self, entry):
        """return Nnew, Nmissing

        Args:
            entry (TYPE): Description

        Returns:
            TYPE: Description
        """
        pdbfile = entry['file']
        cachefile = os.path.join(self.cachedir, 'coord', flatten_path(pdbfile))
        posefile = self.posefile(pdbfile)
        if os.path.exists(cachefile):
            with open(cachefile, 'rb') as f:
                bbstate = list(pickle.load(f))
                if isinstance(bbstate[10], list):
                    bbstate[10] = np.array(bbstate[10], dtype='i4')
                bblock = _BBlock(*bbstate)
            if self.load_poses:
                assert self.load_cached_pose_into_memory(pdbfile)
            self.cache[pdbfile] = bblock
            if self.load_poses:
                self.poses[pdbfile] = pose
            return None, None  # new, missing
        elif self.read_new_pdbs:
            read_pdb = False
            info('PDBPile.build_pdb_data reading %s' % pdbfile)
            if os.path.exists(posefile):
                with open(posefile, 'rb') as f:
                    try:
                        pose = pickle.load(f)
                    except EOFError:
                        print('WARNING corrupt pickled pose', posefile)
                        os.remove(posefile)
            if 'pose' not in vars():
                if not os.path.exists(pdbfile):
                    print("WARNING can't read", pdbfile)
                    return None, pdbfile
                pose = pose_from_file(pdbfile)
                read_pose = True

            ss = Dssp(pose).get_dssp_secstruct()
            bblock = BBlock(entry, pdbfile, pose, ss)
            self.cache[pdbfile] = bblock

            with open(cachefile, 'wb') as f:
                pickle.dump(bblock._state, f)
            if not os.path.exists(posefile):
                with open(posefile, 'wb') as f:
                    pickle.dump(pose, f)
                    info('dumped cache files for %s' % pdbfile)

            if self.load_poses:
                self.poses[pdbfile] = pose
            return pdbfile, None  # new, missing
        else:
            print('no cached data for', pdbfile)
            return None, pdbfile  # new, missing


if __name__ == '__main__':
    import argparse
    import pyrosetta
    info('sent to info')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dbfiles', type=str, nargs='+', dest='database_files')
    parser.add_argument('--nprocs', type=int, dest='nprocs', default=1)
    parser.add_argument(
        '--read_new_pdbs', type=bool, dest='read_new_pdbs', default=False)
    args = parser.parse_args()
    pyrosetta.init('-mute all -ignore_unrecognized_res')

    try:
        pp = PDBPile(
            bakerdb_files=args.database_files,
            nprocs=args.nprocs,
            read_new_pdbs=args.read_new_pdbs,
            metaonly=False,
        )
        print('new entries', pp.n_new_entries)
        print('missing entries', pp.n_missing_entries)
        print('total entries', len(pp.cache))
    except AssertionError as e:
        print(e)
    except:
        if args.read_new_pdbs:
            os.remove(os.environ['HOME'] + '/.worms/cache/lock')
        raise
