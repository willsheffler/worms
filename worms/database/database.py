'''
stores structures and info about categories and splicing and stuff
does a bunch of caching.. maybe too much
'''

import os, json, sys, logging, time, pickle, functools, collections
import concurrent.futures as cf, itertools as it, numpy as np
from logging import info, warning, error

from deferred_import import deferred_import

import worms
import willutil as wu

Databases = collections.namedtuple('Databases', ('bblockdb', 'splicedb'))

class BBlockDatabaseSuper:
   def __init__(self, **kw):
      self.kw = wu.Bunch(kw)

def flatten_path(pdbfile):
   if isinstance(pdbfile, bytes):
      pdbfile = str(pdbfile, 'utf-8')
   return pdbfile.replace(os.sep, '__') + '.pickle'

def read_bblock_dbfiles(dbfiles, dbroot=''):
   dbentries = []
   pdb_contents = dict()
   for dbfile in dbfiles:
      if dbfile.endswith('.txz'):
         arc = worms.database.archive.read_bblock_archive(dbfile)
         dbentries.extend(arc.bblocks)
         pdb_contents.update(arc.pdbs)
      else:
         with open(dbfile) as f:
            dbentries.extend(json.load(f))
   return parse_bblock_database_entries(dbentries, dbroot, pdb_contents)

def parse_bblock_database_entries(dbentries, dbroot='', pdb_contents=dict()):
   for entry in dbentries:
      if 'name' not in entry:
         entry['name'] = ''
      entry['file'] = entry['file'].replace('__PDBDIR__', worms.data.pdb_dir)

   dictdb = {e['file']: e for e in dbentries}
   key_to_pdbfile = {worms.util.hash_str_to_int(e['file']): e['file'] for e in dbentries}

   assert len(dbentries), 'no db entries'
   pdb_files_missing = False
   for entry in dbentries:
      # print('database.py checking', entry['file'])
      if not (os.path.exists(dbroot + entry['file']) or entry['file'] in pdb_contents):
         pdb_files_missing = True
         # print('!' * 60)
         print('pdb file pdb_files_missing:', entry['file'])
         # print('!' * 60)
   assert not pdb_files_missing
   return dbentries, dictdb, key_to_pdbfile, pdb_contents

def get_cachedirs(cachedirs):
   cachedirs = cachedirs or []
   if not isinstance(cachedirs, str):
      cachedirs = [x for x in cachedirs if x]
   if not cachedirs:
      if 'HOME' in os.environ:
         cachedirs = [
            os.environ['HOME'] + os.sep + '.worms/cache',
            '/databases/worms',
         ]
      else:
         cachedirs = ['.worms/cache', '/databases/worms']
   if isinstance(cachedirs, str):
      cachedirs = [cachedirs]
   return cachedirs

def sanitize_pdbfile(pdbfile):
   if isinstance(pdbfile, bytes):
      pdbfile = str(pdbfile, 'utf-8')
   if isinstance(pdbfile, np.ndarray):
      pdbfile = str(bytes(pdbfile), 'utf-8')
   return pdbfile
