import os, json, random, sys, logging, time, pickle, functools
import concurrent.futures as cf, numpy as np
# from logging import info, warning, error
from tqdm import tqdm

import worms
from worms import PING
import willutil as wu

from worms.util.util import get_props_from_url
# logging.basicConfig(level=logging.INFO)

class CachingBBlockDB:
   '''stores Poses and BBlocks in a disk cache'''
   def __str__(self):
      return os.linesep.join([
         f'CachingBBlockDB',
         f'   entries {len(self._alldb)}',
         f'   bblocks {len(self._bblock_cache)}',
         f'   poses {len(self._poses_cache)}',
         f'   dbroot {self.dbroot}',
         f'   cachedirs {self.cachedirs}',
         f'   load_poses {self.load_poses}',
         f'   nprocs {self.nprocs}',
         f'   lazy {self.lazy}',
         f'   read_new_pdbs {self.read_new_pdbs}',
         f'   dbfiles {self.dbfiles}',
         f'   n_missing_entries {self.n_missing_entries}',
         f'   n_new_entries {self.n_new_entries}',
         f'   holding_lock {self._holding_lock}',
      ])

   def __init__(
      self,
      cachedirs=None,
      dbfiles=[],
      load_poses=False,
      nprocs=1,
      lazy=True,
      read_new_pdbs=False,
      verbosity=0,
      dbroot='',
      null_base_names=[],
      **kw,
   ):
      '''Stores building block structures and ancillary data
      '''
      self.null_base_names = null_base_names
      self.cachedirs = worms.database.get_cachedirs(cachedirs)
      self.dbroot = dbroot + '/' if dbroot and not dbroot.endswith('/') else dbroot
      PING(f'CachingBBlockDB cachedirs: {self.cachedirs}')
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
      self.dbfiles = dbfiles
      PING(f'database.py: read database files from {self.dbroot}')
      (
         self._alldb,
         self._dictdb,
         self._key_to_pdbfile,
         self.pdb_contents,
      ) = worms.database.read_bblock_dbfiles(
         dbfiles,
         self.dbroot,
      )
      if len(self._alldb) != len(self._dictdb):
         dups = len(self._alldb) - len(self._dictdb)
         print('WARNING: %6i duplicate pdb files in database' % dups)
      PING('loading %i db entries' % len(self._alldb))
      self.n_new_entries = 0
      self.n_missing_entries = len(self._alldb)
      if not self.lazy:
         self.n_new_entries, self.n_missing_entries = self.load_from_pdbs()
         if self._holding_lock:
            self.unlock_cachedir()
         if nprocs != 1:
            # reload because processpool cache entries not serialized back
            self.nprocs, tmp = 1, self.nprocs
            self.load_from_pdbs()
            # self.nprocs = tmp
      for i, k in enumerate(sorted(self._dictdb)):
         self._alldb[i] = self._dictdb[k]

   def report(self):
      print('CachingBBlockDB nentries:', len(self._alldb))
      print('    dbfiles:', '        \n'.join(self.dbfiles))

   def clear(self):
      self._bblock_cache.clear()
      self._poses_cache.clear()
      if self._holding_lock:
         self.unlock_cachedir()

   def clear_bblocks(self):
      self._bblock_cache.clear()
      if self._holding_lock:
         self.unlock_cachedir()

   def get_json_entry(self, file):
      return self._dictdb[file]

   def acquire_cachedir_lock(self, timeout=20):
      for i in range(timeout):
         if self.islocked_cachedir():
            if i % 10 == 0:
               print(f'waiting {i}/600s to acquire:', os.path.exists(self.cachedirs[0] + '/lock'))
            time.sleep(1)
         else:
            self.lock_cachedir()
            return True
      return False

   def lock_cachedir(self):
      assert not os.path.exists(self.cachedirs[0] + '/lock'), (
         "database is locked! if you're sure no other jobs are editing it, remove " +
         self.cachedirs[0] + '/lock')
      print('locking database', self.cachedirs[0] + '/lock')
      sys.stdout.flush()
      open(self.cachedirs[0] + '/lock', 'w').close()
      assert os.path.exists(self.cachedirs[0] + '/lock')
      self._holding_lock = True

   def unlock_cachedir(self):
      print('unlocking database', self.cachedirs[0] + '/lock')
      sys.stdout.flush()
      os.remove(self.cachedirs[0] + '/lock')
      self._holding_lock = False

   def islocked_cachedir(self):
      return os.path.exists(self.cachedirs[0] + '/lock')

   def check_lock_cachedir(self):
      if not self._holding_lock:
         self.lock_cachedir()

   # def __getitem__(self, i):
   #    if isinstance(i, str):
   #        return self._bblock_cache[i]
   #    else:
   #        return self._bblock_cache.values()[i]

   def __len__(self):
      return len(self._bblock_cache)

   def pose(self, pdbfile, **kw):
      '''load pose from _bblock_cache, read from file if not in memory'''
      import worms
      pdbfile = worms.database.sanitize_pdbfile(pdbfile)
      props = None
      if '?' in pdbfile:
         props = get_props_from_url(pdbfile)
         pdbfile, _ = pdbfile.split('?')
         # print('PDBFILE PROPS')
         # print(pdbfile)
         # print(props, flush=True)
         # assert 0

      if not pdbfile in self._poses_cache:
         if not self.load_cached_pose_into_memory(pdbfile):
            import worms.rosetta_init
            if pdbfile in self.pdb_contents:
               contents = self.pdb_contents[pdbfile]
               self._poses_cache[pdbfile] = worms.rosetta_init.pose_from_str(contents)
            else:
               pdbpath = os.path.join(self.dbroot, pdbfile)
               pose = worms.rosetta_init.pose_from_file(pdbpath)
               self._poses_cache[pdbfile] = pose

               print('!!!!!!!!!!!!!!!!!!')
               print('pdbpath', pdbpath)
               print('pdbfile', pdbfile)
               print('props')
               print(pdbfile, flush=True)
               # assert 0
      if props:
         print('PDBFILE PROPS')
         print(pdbfile)
         print(props, flush=True)

         bblock = worms.bblock.BBlock(self.bblock(pdbfile, **kw))
         start, period = (bblock).repeat_spacing
         pose0 = self._poses_cache[pdbfile]
         nrepeats = props['nrepeats']
         pose = worms.bblock.add_repeat_to_pose(pose0, nrepeats, start, period)
         return pose
      else:
         return self._poses_cache[pdbfile]

   def savepose(self, pdbfile):
      pdbfile = worms.database.sanitize_pdbfile(pdbfile)
      assert pdbfile in self._poses_cache
      fname = self.posefile(pdbfile)
      if not os.path.exists(fname):
         with open(fname, 'wb') as out:
            pickle.dump(self._poses_cache[pdbfile], out)

   def bblock(self, pdbkey, **kw):
      pdburl = None
      if isinstance(pdbkey, (str, bytes)):
         pdburl = pdbkey
         pdbkey = worms.util.hash_str_to_int(pdbkey)
      if isinstance(pdbkey, int):
         if not pdbkey in self._bblock_cache:
            if not self.load_cached_bblock_into_memory(pdbkey, pdburl=pdburl, **kw):
               pdbfile = self._key_to_pdbfile[pdbkey]
               raise ValueError('no bblock data for key', pdbkey, pdbfile, 'in', self.cachedirs)
         return self._bblock_cache[pdbkey]
      elif isinstance(pdbkey, list):
         return [self.bblock(f) for f in pdbkey]
      else:
         raise ValueError('bad pdbkey' + str(type(pdbkey)))

   def all_bblocks(self):
      names = list(db['file'] for db in self._alldb)
      return list(self.bblock(name) for name in names)

   def query(
      self,
      query,
      *,
      useclass=True,
      max_bblocks=150,
      shuffle_bblocks=True,
      parallel=0,
      **kw,
   ):
      names = self.query_names(query, useclass=useclass)
      if len(names) > max_bblocks:
         if shuffle_bblocks:
            random.shuffle(names)
         names = names[:max_bblocks]

      try:
         return [self.bblock(n) for n in names]
      except ValueError:
         self.load_pdbs_multiprocess(names, parallel=parallel)
         return [self.bblock(n) for n in names]

   def query_names(self, query, *, useclass=True, exclude_bases=None):
      return worms.database.query_bblocks(self, query, useclass=useclass,
                                          exclude_bases=exclude_bases)

   def clear_caches(self):
      data = self._poses_cache, self._bblock_cache
      self._poses_cache.clear()
      self._bblock_cache.clear()
      return data

   def restore_caches(self, data):
      self._poses_cache, self._bblock_cache = data

   def load_cached_pose_into_memory(self, pdbfile):

      from worms import rosetta_init

      posefile = self.posefile(pdbfile)
      # print('load_cached_pose_into_memory', pdbfile, flush=True)

      try:
         with open(posefile, 'rb') as f:
            try:
               self._poses_cache[pdbfile] = pickle.load(f)
               return True
            except (EOFError, RuntimeError):
               PING('corrupt pickled pose will be replaced', posefile)
               os.remove(posefile)
               return False
      except (OSError, FileNotFoundError):
         return False

   def bblockfile(self, pdbkey):
      assert not isinstance(pdbkey, str)
      for d in self.cachedirs:
         candidate = os.path.join(d, 'bblock', '%016x.pickle' % pdbkey)
         if os.path.exists(candidate):
            # print('bblockfile', candidate)
            return candidate
      return os.path.join(self.cachedirs[0], 'bblock', '%016x.pickle' % pdbkey)

   def load_cached_bblock_into_memory(
      self,
      pdbkey,
      cache_replace=True,
      pdburl=None,
      **kw,
   ):
      assert not isinstance(pdbkey, (str, bytes))

      if not isinstance(pdbkey, (int, str)):
         success = True
         for f in pdbkey:
            success &= self.load_cached_bblock_into_memory(f)
         return success

         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      bblockfile = self.bblockfile(pdbkey)
      # print('bbfile', bblockfile)
      if not os.path.exists(bblockfile):
         print(f'WARNING: bblock cachefile not found {bblockfile}')
         return False

      with open(bblockfile, 'rb') as f:
         bbstate = list(pickle.load(f))
         entry = self._dictdb[self._key_to_pdbfile[pdbkey]]
         newjson = json.dumps(entry).encode()
         assert bytes(bbstate[0]) == newjson

         if bytes(bbstate[0]) == newjson:

            bblock = worms.bblock._BBlock(*bbstate)

            if pdburl:
               #!!!!!!!!!!!!!!!!!!!
               props = get_props_from_url(pdburl)
               pdburl = pdburl.split('?')[0]
               if 'nrepeats' in props:
                  print('adding extension to bblock')
                  bblock = worms.bblock.make_extended_bblock(bblock, nrepeats=props.nrepeats)

            self._bblock_cache[pdbkey] = bblock

            return True
         print('!!! database entry updated for key', pdbkey, entry['file'])
      if cache_replace:
         print('    removing cachefile', bblockfile)
         os.remove(bblockfile)
         print('    reloading info cache', entry['file'])
         self.load_pdbs_multiprocess([entry['file']], parallel=0)
         return self.load_cached_bblock_into_memory(pdbkey, cache_replace=False)

      return False

   def posefile(self, pdbfile):
      for d in self.cachedirs:
         candidate = os.path.join(d, 'poses', worms.database.flatten_path(pdbfile))
         if os.path.exists(candidate):
            return candidate
      return os.path.join(self.cachedirs[0], 'poses', worms.database.flatten_path(pdbfile))

   def load_pdbs_multiprocess(self, names, parallel=0):

      self.read_new_pdbs, tmp = True, self.read_new_pdbs
      data = self.clear_caches()
      needs_unlock = False
      if not self._holding_lock:
         needs_unlock = True
         if not self.acquire_cachedir_lock():
            raise ValueError('cachedir locked, cant write new entries.\n'
                             'If no other worms jobs are running, you may manually remove:\n' +
                             self.cachedirs[0] + '/lock')
      exe = worms.util.InProcessExecutor()
      if parallel:
         exe = cf.ProcessPoolExecutor(max_workers=parallel)
      with exe as pool:
         futures = list()
         for n in names:
            futures.append(pool.submit(self.build_pdb_data, self._dictdb[n], uselock=False))
         iter = cf.as_completed(futures)
         iter = tqdm(iter, 'loading pdb files', total=len(futures))
         for f in iter:
            f.result()
      if needs_unlock:
         self.unlock_cachedir()
      self.restore_caches(data)
      self.read_new_pdbs = tmp

   def load_from_pdbs(self):
      random.shuffle(self._alldb)
      if self.nprocs == 1:
         with worms.util.InProcessExecutor() as exe:
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
      random.shuffle(self._alldb)
      r = []
      kwargs = {
         'total': len(self._alldb),
         'unit': 'pdbs',
         # 'unit_scale': True,
         'leave': True,
      }
      futures = [exe.submit(self.build_pdb_data, e) for e in self._alldb]
      work = cf.as_completed(futures)
      if self.verbosity > 1:
         work = tqdm(work, 'building pdb data', **kwargs)
      for f in work:
         r.append(f.result())
      return r

   def build_pdb_data(self, entry, uselock=True):
      '''return Nnew, Nmissing'''
      # print('build_pdb_data', entry['file'])
      pdbfile = entry['file']
      pdbkey = worms.util.hash_str_to_int(pdbfile)
      cachefile = self.bblockfile(pdbkey)
      posefile = self.posefile(pdbfile)

      if os.path.exists(cachefile):
         if not self.load_cached_bblock_into_memory(pdbkey):
            if os.path.exists(cachefile):
               raise ValueError(
                  f'cachefile {cachefile} exists, but cant load data from associated key {pdbkey}')
            raise ValueError(
               f'cachefile {cachefile} was removed, cant load data from associated key {pdbkey}')
         if self.load_poses:
            if not self.load_cached_pose_into_memory(pdbfile):
               print('warning, not saved:', pdbfile)
         return None, None  # new, missing
      elif self.read_new_pdbs:

         if uselock:
            self.check_lock_cachedir()
         read_pdb = False
         # info('CachingBBlockDB.build_pdb_data reading %s' % pdbfile)

         pose = self.pose(pdbfile)

         bblock = worms.bblock.make_bblock(entry, pose, self.null_base_names)
         self._bblock_cache[pdbkey] = bblock

         with open(cachefile, 'wb') as f:
            pickle.dump(bblock._state, f)
         # print('saved new bblock cache file', cachefile)
         if not os.path.exists(posefile):
            try:
               with open(posefile, 'wb') as f:
                  pickle.dump(pose, f)
                  PING('dumped _bblock_cache files for %s' % pdbfile)
            except OSError as e:
               print('not saving', posefile)

         if self.load_poses:
            self._poses_cache[pdbfile] = pose
         return pdbfile, None  # new, missing
      else:
         print('warning no cached data for: ' + pdbfile)
         return None, pdbfile  # new, missing
