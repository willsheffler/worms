"""
stores structures and info about categories and splicing and stuff
does a bunch of caching.. maybe too much
"""

import os, json, sys, logging, time, pickle, functools, collections
import concurrent.futures as cf, itertools as it, numpy as np
from logging import info, warning, error

from deferred_import import deferred_import

import worms

Databases = collections.namedtuple('Databases', ('bblockdb', 'splicedb'))

def flatten_path(pdbfile):
   if isinstance(pdbfile, bytes):
      pdbfile = str(pdbfile, "utf-8")
   return pdbfile.replace(os.sep, "__") + ".pickle"

def read_bblock_dbfiles(dbfiles, dbroot=''):
   alldb = []
   for dbfile in dbfiles:
      with open(dbfile) as f:
         try:
            alldb.extend(json.load(f))
         except json.decoder.JSONDecodeError as e:
            print("ERROR on json file:", dbfile)
            print(e)
            sys.exit()
   for entry in alldb:
      if "name" not in entry:
         entry["name"] = ""
      entry["file"] = entry["file"].replace("__PDBDIR__", worms.data.pdb_dir)

   dictdb = {e["file"]: e for e in alldb}
   key_to_pdbfile = {worms.util.hash_str_to_int(e["file"]): e["file"] for e in alldb}

   assert len(alldb), 'no db entries'
   pdb_files_missing = False
   for entry in alldb:
      if not os.path.exists(dbroot + entry["file"]):
         pdb_files_missing = True
         print('!' * 60)
         print("pdb file pdb_files_missing:", entry["file"])
         print('!' * 60)
   assert not pdb_files_missing
   return alldb, dictdb, key_to_pdbfile

def get_cachedirs(cachedirs):
   cachedirs = cachedirs or []
   if not isinstance(cachedirs, str):
      cachedirs = [x for x in cachedirs if x]
   if not cachedirs:
      if "HOME" in os.environ:
         cachedirs = [
            os.environ["HOME"] + os.sep + ".worms/cache",
            "/databases/worms",
         ]
      else:
         cachedirs = [".worms/cache", "/databases/worms"]
   if isinstance(cachedirs, str):
      cachedirs = [cachedirs]
   return cachedirs

def sanitize_pdbfile(pdbfile):
   if isinstance(pdbfile, bytes):
      pdbfile = str(pdbfile, "utf-8")
   if isinstance(pdbfile, np.ndarray):
      pdbfile = str(bytes(pdbfile), "utf-8")
   return pdbfile
