"""TODO: Summary
"""
import os
import json
import random
import sys
import concurrent.futures as cf
import itertools as it
import logging
from logging import info, warning, error
from random import shuffle
import time

import numpy as np
from tqdm import tqdm

from worms.util import hash_str_to_int
from worms import util
from worms.bblock import BBlock, _BBlock

logging.basicConfig(level=logging.INFO)
import _pickle as pickle

try:
    # god, I'm so tired of this crap....
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
    HAVE_PYROSETTA = True
except ImportError:
    HAVE_PYROSETTA = False


def flatten_path(pdbfile):
    if isinstance(pdbfile, bytes):
        pdbfile = str(pdbfile, 'utf-8')
    return pdbfile.replace(os.sep, '__') + '.pickle'


class SpliceDB:
    """Stores valid NC splices for bblock pairs"""

    def __init__(self, cachedirs=None, **kw):
        cachedirs = cachedirs or []
        if not isinstance(cachedirs, str):
            cachedirs = [x for x in cachedirs if x]
        if not cachedirs:
            if 'HOME' in os.environ:
                cachedirs = os.environ['HOME'] + os.sep + '.worms/cache'
            else:
                cachedirs = ['.worms/cache']
        if isinstance(cachedirs, str):
            cachedirs = [cachedirs]
        self.cachedirs = [os.path.join(x, 'splices') for x in cachedirs]
        self._cache = dict()
        self._dirty = set()

    def partial(self, params, pdbkey):
        assert isinstance(pdbkey, int)
        if (params, pdbkey) not in self._cache:
            cachefile = self.cachepath(params, pdbkey)
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    self._cache[params, pdbkey] = pickle.load(f)
            else:
                self._cache[params, pdbkey] = dict()
        return self._cache[params, pdbkey]

    def has(self, params, pdbkey0, pdbkey1):
        k = (params, pdbkey0)
        if k in self._cache:
            if pdbkey1 in self._cache[k]:
                return True
        return False

    def add(self, params, pdbkey0, pdbkey1, val):
        assert isinstance(pdbkey0, int)
        assert isinstance(pdbkey1, int)
        k = (params, pdbkey0)
        self._dirty.add(k)
        if k not in self._cache: self._cache[k] = dict()
        self._cache[k][pdbkey1] = val

    def cachepath(self, params, pdbkey):
        # stock hash ok for tuples of numbers (?)
        prm = '%016x' % abs(hash(params))
        key = '%016x.pickle' % pdbkey
        for d in self.cachedirs:
            candidate = os.path.join(d, prm, key)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.cachedirs[0], prm, key)

    def listpath(self, params, pdbkey):
        return self.cachepath(params,
                              pdbkey).replace('.pickle', '_list.pickle')

    def sync_to_disk(self, dirty_only=True):
        for i in range(10):
            keys = list(self._dirty) if dirty_only else self.cache.keys()
            for key in keys:
                cachefile = self.cachepath(*key)
                if os.path.exists(cachefile + '.lock'):
                    continue
                if not os.path.exists(os.path.dirname(cachefile)):
                    os.makedirs(os.path.dirname(cachefile))
                with open(cachefile + '.lock', 'w'):
                    if os.path.exists(cachefile):
                        with open(cachefile, 'rb') as inp:
                            self._cache[key].update(pickle.load(inp))
                    with open(cachefile, 'wb') as out:
                        data = self._cache[key]
                        pickle.dump(data, out)
                    listfile = self.listpath(*key)
                    with open(listfile, 'wb') as out:
                        data = set(self._cache[key].keys())
                        pickle.dump(data, out)
                os.remove(cachefile + '.lock')
                self._dirty.remove(key)
        if len(self._dirty):
            print(self._dirty)
            print('warning: some caches unsaved', len(self._dirty))


class BBlockDB:
    """stores Poses and BBlocks in a disk cache"""

    def __init__(
            self,
            cachedirs=None,
            dbfiles=[],
            load_poses=False,
            nprocs=1,
            lazy=True,
            read_new_pdbs=False,
            verbosity=0,
            **kw
    ):
        """TODO: Summary

        Args:
            cachedirs (None, optional): Description
            dbfiles (list, optional): Description
            load_poses (bool, optional): Description
            nprocs (int, optional): Description
            lazy (bool, optional): Description
            read_new_pdbs (bool, optional): Description
        """
        cachedirs = cachedirs or []
        if not isinstance(cachedirs, str):
            cachedirs = [x for x in cachedirs if x]
        if not cachedirs:
            if 'HOME' in os.environ:
                cachedirs = [os.environ['HOME'] + os.sep + '.worms/cache']
            else:
                cachedirs = ['.worms/cache']
        if isinstance(cachedirs, str):
            cachedirs = [cachedirs]
        self.cachedirs = cachedirs
        self.load_poses = load_poses
        os.makedirs(self.cachedirs[0] + '/poses', exist_ok=True)
        os.makedirs(self.cachedirs[0] + '/bblock', exist_ok=True)
        self._bblock_cache, self._poses_cache = dict(), dict()
        self.nprocs = nprocs
        self.lazy = lazy
        self.read_new_pdbs = read_new_pdbs
        self.verbosity = verbosity
        self._alldb = []
        self._holding_lock = False
        for dbfile in dbfiles:
            with open(dbfile) as f:
                self._alldb.extend(json.load(f))
        for entry in self._alldb:
            if 'name' not in entry:
                entry['name'] = ''
            entry['file'] = entry['file'].replace(
                '__DATADIR__',
                os.path.relpath(os.path.dirname(__file__) + '/data')
            )
        self._dictdb = {e['file']: e for e in self._alldb}
        self.key_to_pdbfile = {
            hash_str_to_int(e['file']): e['file']
            for e in self._alldb
        }
        if len(self._alldb) != len(self._dictdb):
            warning('!' * 100)
            warning(
                '!' * 23, 'DIRE WARNING: %6i duplicate pdb files in database' %
                (len(self._alldb) - len(self._dictdb)), '!' * 23
            )
            warning('!' * 100)
        info('loading %i db entries' % len(self._alldb))
        self.n_new_entries = 0
        self.n_missing_entries = len(self._alldb)
        if not self.lazy:
            self.n_new_entries, self.n_missing_entries = self.load_from_pdbs()
            if self._holding_lock: self.unlock_cachedir()
            if nprocs != 1:
                # reload because processpool cache entries not serialized back
                self.nprocs = 1
                self.load_from_pdbs()
        for i, k in enumerate(sorted(self._dictdb)):
            self._alldb[i] = self._dictdb[k]

    def clear(self):
        self._bblock_cache, self._poses_cache = dict(), dict()

    def acquire_cachedir_lock(self, timeout=600):
        for i in range(timeout):
            if self.islocked_cachedir():
                if i % 10 == 0:
                    print(f'waiting {i}/600s to acquire_cachedir_lock')
                time.sleep(1)
            else:
                self.lock_cachedir()
                return True
        return False

    def lock_cachedir(self):
        assert not os.path.exists(self.cachedirs[0] + '/lock'), (
            "database is locked! if you're sure no other jobs are editing it, remove "
            + self.cachedirs[0] + "/lock"
        )
        open(self.cachedirs[0] + '/lock', 'w').close()
        assert os.path.exists(self.cachedirs[0] + '/lock')
        self._holding_lock = True

    def unlock_cachedir(self):
        os.remove(self.cachedirs[0] + '/lock')
        self._holding_lock = False

    def islocked_cachedir(self):
        return os.path.exists(self.cachedirs[0] + '/lock')

    def check_lock_cachedir(self):
        if not self._holding_lock:
            self.lock_cachedir()

    #def __getitem__(self, i):
    #    if isinstance(i, str):
    #        return self._bblock_cache[i]
    #    else:
    #        return self._bblock_cache.values()[i]

    def __len__(self):
        return len(self._bblock_cache)

    def pose(self, pdbfile):
        """load pose from _bblock_cache, read from file if not in memory"""
        if isinstance(pdbfile, bytes):
            pdbfile = str(pdbfile, 'utf-8')
        if isinstance(pdbfile, np.ndarray):
            pdbfile = str(bytes(pdbfile), 'utf-8')
        if not pdbfile in self._poses_cache:
            if not self.load_cached_pose_into_memory(pdbfile):
                self._poses_cache[pdbfile] = pose_from_file(pdbfile)
        return self._poses_cache[pdbfile]

    def bblock(self, pdbkey):
        if isinstance(pdbkey, (str, bytes)):
            pdbkey = hash_str_to_int(pdbkey)
        if isinstance(pdbkey, int):
            if not pdbkey in self._bblock_cache:
                if not self.load_cached_bblock_into_memory(pdbkey):
                    pdbfile = self.key_to_pdbfile[pdbkey]
                    raise ValueError(
                        'no bblock data for key', pdbkey, pdbfile, 'in',
                        self.cachedirs
                    )
            return self._bblock_cache[pdbkey]
        elif isinstance(pdbkey, list):
            return [self.bblock(f) for f in pdbkey]
        else:
            raise ValueError('bad pdbkey' + str(type(pdbkey)))

    def query(
            self,
            query,
            *,
            useclass=True,
            max_bblocks=150,
            shuffle_bblocks=True,
            parallel=0,
            **kw
    ):
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
        names = self.query_names(query, useclass=useclass, **kw)
        if len(names) > max_bblocks:
            if shuffle_bblocks: random.shuffle(names)
            names = names[:max_bblocks]

        try:
            return [self.bblock(n) for n in names]
        except ValueError:
            self.load_pdbs_multiprocess(names, parallel)
            return [self.bblock(n) for n in names]

    def query_names(self, query, *, useclass=True, exclude_bases=None):
        """query for names only"""
        if query.lower() == "all":
            return [db['file'] for db in self._alldb]
        query, subq = query.split(':') if query.count(':') else (query, None)
        if subq is None:
            c_hits = [db['file'] for db in self._alldb if query in db['class']]
            n_hits = [db['file'] for db in self._alldb if query == db['name']]
            t_hits = [db['file'] for db in self._alldb if query == db['type']]
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
            for db in self._alldb:
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
        if exclude_bases is not None:
            hits0, hits = hits, []
            for h in hits0:
                base = self._dictdb[h]['base']
                if base == '' or base not in exclude_bases:
                    hits.append(h)
            print('exclude_bases', len(hits0), len(hits))
        return hits

    def clear_caches(self):
        data = self._poses_cache, self._bblock_cache
        self._poses_cache = dict()
        self._bblock_cache = dict()
        return data

    def restore_caches(self, data):
        self._poses_cache, self._bblock_cache = data

    def load_cached_pose_into_memory(self, pdbfile):
        posefile = self.posefile(pdbfile)
        try:
            with open(posefile, 'rb') as f:
                try:
                    self._poses_cache[pdbfile] = pickle.load(f)
                    return True
                except EOFError:
                    warning('corrupt pickled pose will be replaced', posefile)
                    os.remove(posefile)
                    return False
        except FileNotFoundError:
            return False

    def bblockfile(self, pdbkey):
        assert not isinstance(pdbkey, str)
        for d in self.cachedirs:
            candidate = os.path.join(d, 'bblock', '%016x.pickle' % pdbkey)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(
            self.cachedirs[0], 'bblock', '%016x.pickle' % pdbkey
        )

    def load_cached_bblock_into_memory(self, pdbkey):
        assert not isinstance(pdbkey, (str, bytes))
        if not isinstance(pdbkey, (int, str)):
            success = True
            for f in pdbkey:
                success &= self.load_cached_bblock_into_memory(f)
            return success
        bblockfile = self.bblockfile(pdbkey)
        try:
            with open(bblockfile, 'rb') as f:
                bbstate = list(pickle.load(f))
                self._bblock_cache[pdbkey] = _BBlock(*bbstate)
                return True
        except FileNotFoundError:
            return False

    def posefile(self, pdbfile):
        for d in self.cachedirs:
            candidate = os.path.join(d, 'poses', flatten_path(pdbfile))
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.cachedirs[0], 'poses', flatten_path(pdbfile))

    def load_pdbs_multiprocess(self, names, parallel):
        self.read_new_pdbs, tmp = True, self.read_new_pdbs
        data = self.clear_caches()
        if not self.acquire_cachedir_lock():
            raise ValueError(
                'cachedir locked, cant write new entries.\n'
                'If no other worms jobs are running, you may manually remove:\n'
                + self.cachedirs[0] + '/lock'
            )
        exe = util.InProcessExecutor()
        if parallel: exe = cf.ProcessPoolExecutor(max_workers=parallel)
        with exe as pool:
            futures = list()
            for n in names:
                futures.append(
                    pool.submit(
                        self.build_pdb_data, self._dictdb[n], uselock=False
                    )
                )
            iter = cf.as_completed(futures)
            iter = tqdm(iter, 'loading pdb files', total=len(futures))
            for f in iter:
                f.result()
        self.unlock_cachedir()
        self.restore_caches(data)
        self.read_new_pdbs = tmp

    def load_from_pdbs(self):
        shuffle(self._alldb)
        if self.nprocs is 1:
            with util.InProcessExecutor() as exe:
                result = self.load_from_pdbs_inner(exe)
        else:
            with cf.ThreadPoolExecutor(max_workers=self.nprocs) as exe:
                result = self.load_from_pdbs_inner(exe)
        new = [_[0] for _ in result if _[0]]
        missing = [_[1] for _ in result if _[1]]
        for miss in missing:
            self._alldb.remove(self._dictdb[miss])
            del self._dictdb[miss]
        return len(new), len(missing)

    def load_from_pdbs_inner(self, exe):
        # return exe.map(self.build_pdb_data, self._alldb)
        shuffle(self._alldb)
        r = []
        kwargs = {
            'total': len(self._alldb),
            'unit': 'pdbs',
            # 'unit_scale': True,
            'leave': True
        }
        futures = [exe.submit(self.build_pdb_data, e) for e in self._alldb]
        work = cf.as_completed(futures)
        if self.verbosity > 1:
            work = tqdm(work, 'building pdb data', **kwargs)
        for f in work:
            r.append(f.result())
        return r

    def build_pdb_data(self, entry, uselock=True):
        """return Nnew, Nmissing"""
        pdbfile = entry['file']
        pdbkey = hash_str_to_int(pdbfile)
        cachefile = self.bblockfile(pdbkey)
        posefile = self.posefile(pdbfile)
        if os.path.exists(cachefile):
            assert self.load_cached_bblock_into_memory(pdbkey)
            if self.load_poses:
                assert self.load_cached_pose_into_memory(pdbfile)
            return None, None  # new, missing
        elif self.read_new_pdbs:
            if uselock: self.check_lock_cachedir()
            read_pdb = False
            # info('BBlockDB.build_pdb_data reading %s' % pdbfile)
            pose = self.pose(pdbfile)
            ss = Dssp(pose).get_dssp_secstruct()
            bblock = BBlock(entry, pdbfile, pdbkey, pose, ss)
            self._bblock_cache[pdbfile] = bblock
            # print(cachefile)
            with open(cachefile, 'wb') as f:
                pickle.dump(bblock._state, f)
            if not os.path.exists(posefile):
                with open(posefile, 'wb') as f:
                    pickle.dump(pose, f)
                    info('dumped _bblock_cache files for %s' % pdbfile)

            if self.load_poses:
                self._poses_cache[pdbfile] = pose
            return pdbfile, None  # new, missing
        else:
            warning('no cached data for: ' + pdbfile)
            return None, pdbfile  # new, missing
