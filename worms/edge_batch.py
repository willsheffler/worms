import os
import sys
from tqdm import tqdm
import concurrent.futures as cf
from collections import defaultdict
import _pickle
from random import shuffle

import numpy as np

from worms.vertex import Vertex
from worms.edge import Edge
from worms.edge import _jit_splice_metrics, _splice_respairs, _analysis
from worms.bblock import BBlock
from worms.util import InProcessExecutor, hash_str_to_int

def _valid_splice_pairs(bbw0, bbw1, **kw):
   blk0 = bbw0._bblock
   blk1 = bbw1._bblock
   rms, nclash, ncontact, ncnh, nhc = _jit_splice_metrics(
      blk0.chains,
      blk1.chains,
      blk0.ncac,
      blk1.ncac,
      blk0.stubs,
      blk1.stubs,
      blk0.connections,
      blk1.connections,
      blk0.ss,
      blk1.ss,
      blk0.cb,
      blk1.cb,
      kw["splice_clash_d2"],
      kw["splice_contact_d2"],
      kw["splice_rms_range"],
      kw["splice_clash_contact_range"],
      kw["splice_clash_contact_by_helix"],
      kw["splice_max_rms"],
      kw["splice_max_chain_length"],
      kw["splice_min_dotz"],
      True,
   )
   ok, analysis = _analysis(nclash, rms, ncontact, ncnh, nhc, **kw)
   return _splice_respairs(ok, blk0, blk1), analysis

def compute_splices(bbdb, bbpairs, verbosity, parallel, pbar, pbar_interval=10.0, **kw):
   bbpairs_shuf = bbpairs.copy()
   shuffle(bbpairs_shuf)
   exe = InProcessExecutor()
   if parallel:
      exe = cf.ProcessPoolExecutor(max_workers=parallel)
   with exe as pool:
      futures = list()
      for bbpair in bbpairs_shuf:
         bbw0 = BBlock(bbdb.bblock(bbpair[0]))
         bbw1 = BBlock(bbdb.bblock(bbpair[1]))
         f = pool.submit(_valid_splice_pairs, bbw0, bbw1, **kw)
         f.bbpair = bbpair
         futures.append(f)
      print("batch compute_splices, npairs:", len(futures))
      fiter = cf.as_completed(futures)
      if pbar:
         fiter = tqdm(fiter, "precache splices", mininterval=pbar_interval, total=len(futures))
      res = {f.bbpair: f.result() for f in fiter}
   return {bbpair: res[bbpair] for bbpair in bbpairs}

def _remove_already_cached(spdb, bbpairs, params):
   pairmap = defaultdict(list)
   for a, b in bbpairs:
      pairmap[a].append(b)
   ret = list()
   for pdb0, pdb1s in pairmap.items():
      pdbkey0 = hash_str_to_int(pdb0)
      if all(spdb.has(params, pdbkey0, hash_str_to_int(p1)) for p1 in pdb1s):
         continue
      listpath = spdb.listpath(params, pdbkey0)
      haveit = set()
      if os.path.exists(listpath):
         with open(listpath, "rb") as inp:
            haveit = _pickle.load(inp)
      for pdb1 in pdb1s:
         pdbkey1 = hash_str_to_int(pdb1)
         if not pdbkey1 in haveit:
            ret.append((pdb0, pdb1))
   return ret

def precompute_splicedb(database, bbpairs, **kw):
   bbdb, spdb = database

   # note: this is duplicated in edge.py and they need to be the same
   params = (
      kw["splice_max_rms"],
      kw["splice_ncontact_cut"],
      kw["splice_clash_d2"],
      kw["splice_contact_d2"],
      kw["splice_rms_range"],
      kw["splice_clash_contact_range"],
      kw["splice_clash_contact_by_helix"],
      kw["splice_ncontact_no_helix_cut"],
      kw["splice_nhelix_contacted_cut"],
      kw["splice_max_chain_length"],
      kw["splice_min_dotz"],
   )
   bbpairs = _remove_already_cached(spdb, bbpairs, params)
   if not bbpairs:
      return

   splices = compute_splices(bbdb, bbpairs, **kw)
   for key, val in splices.items():
      pdbkey0 = hash_str_to_int(key[0])
      pdbkey1 = hash_str_to_int(key[1])
      spdb.add(params, pdbkey0, pdbkey1, val)

   spdb.sync_to_disk()

   print("precompute_splicedb done")
   sys.stdout.flush()
