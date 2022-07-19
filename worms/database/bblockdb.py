import os, random, functools, pickle
import numpy as np
import worms
import willutil as wu

class BBlockDB(worms.database.BBlockDatabaseSuper):
   def __init__(
      self,
      dbfiles=[],
      cachedirs=[],
      dbroot='',
      null_base_names=['', '?', 'n/a', 'none'],
      **kw,
   ):
      super().__init__(**kw)
      self.dbfiles = dbfiles
      self.dbroot = dbroot + "/" if dbroot and not dbroot.endswith("/") else dbroot
      (
         self._alldb,
         self._dictdb,
         self._key_to_pdbfile,
         self.pdb_contents,
      ) = worms.database.read_bblock_dbfiles(
         dbfiles,
         self.dbroot,
      )
      self.cachedirs = worms.database.get_cachedirs(cachedirs)
      self._bblock_cache = dict()
      self.null_base_names = null_base_names
      self.bblocks_accessed = set()
      self.poses_accessed = set()
      self.kw = wu.Bunch(kw)

   def merge_into_self(self, other, keep_access_info=False):
      self.dbfiles = list({*self.dbfiles, *other.dbfiles})
      assert self.dbroot == other.dbroot
      assert self.cachedirs == other.cachedirs
      self._bblock_cache.update(other._bblock_cache)
      assert self.null_base_names == other.null_base_names
      self._dictdb.update(other._dictdb)
      self._alldb = list(self._dictdb.values())
      self._key_to_pdbfile.update(other._key_to_pdbfile)
      if keep_access_info:
         self.bblocks_accessed.update(other.bblocks_accessed)
         self.poses_accessed.update(other.poses_accessed)
      else:
         self.bblocks_accessed = set()
         self.poses_accessed = set()

   def __setstate__(self, state):
      assert len(state) == 8
      self.dbfiles = state[0]
      self.dbroot = state[1]
      self.cachedirs = state[2]
      self._bblock_cache = {k: worms.bblock._BBlock(*v) for k, v in state[3].items()}
      self.null_base_names = state[4]
      self._alldb = state[5]
      self._dictdb = state[6]
      self._key_to_pdbfile = state[7]
      self.bblocks_accessed = set()
      self.poses_accessed = set()

   def __getstate__(self):
      return (
         self.dbfiles,
         self.dbroot,
         self.cachedirs,
         {k: v._state for k, v in self._bblock_cache.items()},
         self.null_base_names,
         self._alldb,
         self._dictdb,
         self._key_to_pdbfile,
      )

   def posefile(self, pdbfile):
      for d in self.cachedirs:
         candidate = os.path.join(d, "poses", worms.database.flatten_path(pdbfile))
         if os.path.exists(candidate):
            return candidate
      return None

   @functools.lru_cache(128)
   def get_pose(self, pdbfile):
      posefile = self.posefile(pdbfile)
      if posefile:
         with open(posefile, "rb") as f:
            return pickle.load(f)
      else:
         print("reading pdb", pdbfile)
         if pdbfile in self.pdb_contents:
            return worms.rosetta_init.pose_from_str(self.pdb_contents[pdbfile])
         else:
            assert os.path.exists(self.dbroot + pdbfile)
            return worms.rosetta_init.pose_from_file(self.dbroot + pdbfile)

   def pose(self, pdbfile):
      """load pose from _bblock_cache, read from file if not in memory. only reads"""
      if isinstance(pdbfile, bytes):
         pdbfile = str(pdbfile, "utf-8")
      if isinstance(pdbfile, np.ndarray):
         pdbfile = str(bytes(pdbfile), "utf-8")
      self.poses_accessed.add(pdbfile)
      return self.get_pose(pdbfile)

   def bblock(self, pdbkey):
      if isinstance(pdbkey, list):
         return [self.bblock(f) for f in pdbkey]
      if isinstance(pdbkey, (str, bytes)):

         import worms  # todo, why is this necessary??

         pdbkey = worms.util.hash_str_to_int(pdbkey)
      assert isinstance(pdbkey, int)
      if not pdbkey in self._bblock_cache:
         import worms.rosetta_init
         pdbfile = self._key_to_pdbfile[pdbkey]
         pose = self.pose(pdbfile)
         entry = self._dictdb[pdbfile]
         ss = worms.rosetta_init.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
         bblock = worms.bblock.make_bblock(entry, pose, self.null_base_names, **self.kw)
         self._bblock_cache[pdbkey] = bblock
      self.bblocks_accessed.add(self._key_to_pdbfile[pdbkey])
      return self._bblock_cache[pdbkey]

   def load_all_bblocks(self):
      allbb = list()
      for i, k in enumerate(self._dictdb):
         if i % 100 == 0:
            print(f'load_all_bblocks progress {i} of {len(self._dictdb)}', flush=True)
         allbb.append(self.bblock(k))  # will load and cache in memory
      return allbb

   def all_bblocks(self):
      return self.load_all_bblocks()

   def loaded_pdbs(self):
      return self._bblock_cache.keys()

   def query(self, query, *, useclass=True, max_bblocks=150, shuffle_bblocks=True, **kw):
      names = self.query_names(query, useclass=useclass)
      if len(names) > max_bblocks:
         if shuffle_bblocks:
            random.shuffle(names)
         names = names[:max_bblocks]
      return [self.bblock(n) for n in names]

   def query_names(self, query, *, useclass=True, exclude_bases=None):
      return worms.database.query_bblocks(self, query, useclass=useclass,
                                          exclude_bases=exclude_bases)

   def get_json_entry(self, file):
      return self._dictdb[file]

   def clear(self):
      self._bblock_cache.clear()
      # self._poses_cache.clear()

   def clear_bblocks(self):
      self._bblock_cache.clear()

   def report(self):
      print("BBlockDB nentries:", len(self._alldb))

   def __str__(self):
      return os.linesep.join([
         f'BBlockDB',
         f'   dbroot: {self.dbroot}',
         f'   null_base_names: {self.null_base_names}',
         f'   entries: {len(self._dictdb)}',
         f'   loaded bblocks: {len(self._bblock_cache)}',
         f'   accessed bblocks: {len(self.bblocks_accessed)}',
         f'   accessed poses: {len(self.poses_accessed)}',
         f'   dbfiles:',
         os.linesep.join([f'      {x}' for x in self.dbfiles]),
      ])
