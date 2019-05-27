import sys
from time import time
from collections import Counter
import concurrent.futures as cf
import _pickle
import os

import numpy as np

import worms
from worms import Vertex, Edge, precompute_splicedb
from worms.bblock import bblock_dump_pdb, _BBlock
from worms.vertex import _Vertex
from worms.edge import _Edge
from worms.util import InProcessExecutor
from pprint import pprint
from logging import info
import string


def _validate_bbs_verts(bbs, verts):
   assert len(bbs) == len(verts)
   for bb, vert in zip(bbs, verts):
      if vert is None:
         continue
      assert 0 <= np.min(vert.ibblock)
      assert np.max(vert.ibblock) < len(bb)


class SearchSpaceDag:
   """represents search space
    """

   def __init__(self, bbspec, bbs, verts, edges):
      _validate_bbs_verts(bbs, verts)
      assert isinstance(bbs[0][0], _BBlock)
      assert isinstance(verts[0], (_Vertex, type(None)))
      if not (len(edges) == 0 or all(isinstance(e, _Edge) for e in edges)):
         raise ValueError("Error bad SearchSpaceDag edges")
      if bbspec:
         assert len(bbspec) == len(bbs)
      assert len(edges) == 0 or len(edges) + 1 == len(verts)
      self.bbspec = bbspec
      self.bbs = tuple(bbs)
      self.verts = tuple(verts)
      self.edges = tuple(edges)

   def __getstate__(self):
      return (
         self.bbspec,
         [[x._state for x in bb] for bb in self.bbs],
         [x._state for x in self.verts],
         [x._state for x in self.edges],
      )

   def __setstate__(self, state):
      self.bbspec = state[0]
      self.bbs = tuple(tuple(_BBlock(*x) for x in bb) for bb in state[1])
      self.verts = tuple(_Vertex(*x) for x in state[2])
      self.edges = tuple(_Edge(*x) for x in state[3])
      _validate_bbs_verts(self.bbs, self.verts)
      assert len(self.bbs) == len(self.verts) == len(self.edges) + 1

   def get_bases(self, idx):
      assert len(idx) == len(self.verts)
      bases = list()
      for i in range(len(idx)):
         bb = self.bbs[i][self.verts[i].ibblock[idx[i]]]
         bases.append(bytes(bb.base).decode("utf-8"))
      return bases

   def get_base_hashes(self, idx):
      assert len(idx) == len(self.verts)
      bases = list()
      for i in range(len(idx)):
         bb = self.bbs[i][self.verts[i].ibblock[idx[i]]]
         bases.append(bb.basehash)
      return bases

   def report_memory_use(self):
      memvert = [x.memuse // 2**10 for x in self.verts]
      print(f"    vertex memuse (kb): {sum(memvert):8,}", memvert)
      memedge = [x.memuse // 2**10 for x in self.edges]
      print(f"    edge memuse (kb):   {sum(memedge):8,}", memedge)

   def report_size(self):
      sizevert = [x.memuse for x in self.verts]
      sizeedge = [x.memuse for x in self.edges]
      print("SearchSpaceDag sizes:")
      print(f"    vertex sizes: {sum(sizevert):8,}", sizevert)
      print(f"    edge sizes:   {sum(sizeedge):8,}", sizeedge)


def simple_search_dag(
      criteria,
      db=None,
      nbblocks=[64],
      min_seg_len=15,
      parallel=False,
      verbosity=0,
      timing=0,
      modbbs=None,
      make_edges=True,
      merge_bblock=None,
      merge_segment=None,
      precache_splices=False,
      precache_only=False,
      bbs=None,
      bblock_ranges=[],
      only_seg=None,
      source=None,
      print_edge_summary=False,
      no_duplicate_bases=False,
      shuffle_bblocks=False,
      use_saved_bblocks=False,
      output_prefix="./worms",
      only_ivertex=[],
      **kw,
):
   bbdb, spdb = db
   queries, directions = zip(*criteria.bbspec)
   tdb = time()
   if bbs is None:
      bbs = list()
      savename = output_prefix + "_bblocks.pickle"

      if use_saved_bblocks and os.path.exists(savename):
         with open(savename, "rb") as inp:
            bbnames_list = _pickle.load(inp)
         # for i, l in enumerate(bbnames_list)
         # if len(l) >= nbblocks[i]:
         # assert 0, f"too many bblocks in {savename}"
         for i, bbnames in enumerate(bbnames_list):
            bbs.append([bbdb.bblock(n) for n in bbnames[:nbblocks[i]]])

      else:
         for iquery, query in enumerate(queries):
            if hasattr(criteria, "cloned_segments"):
               msegs = [i + len(queries) if i < 0 else i for i in criteria.cloned_segments()]
               if iquery in msegs[1:]:
                  print("seg", iquery, "repeating bblocks from", msegs[0])
                  bbs.append(bbs[msegs[0]])
                  continue
            bbs0 = bbdb.query(
               query,
               max_bblocks=nbblocks[iquery],
               shuffle_bblocks=shuffle_bblocks,
               parallel=parallel,
            )
            bbs.append(bbs0)

         if bblock_ranges:
            bbs_sliced = list()
            assert len(bblock_ranges) == 2 * len(bbs)
            for ibb, bb in enumerate(bbs):
               lb, ub = bblock_ranges[2 * ibb:2 * ibb + 2]
               bbs_sliced.append(bb[lb:ub])
            bbs = bbs_sliced

         for ibb, bb in enumerate(bbs):
            print("bblocks", ibb)
            for b in bb:
               print("   ", bytes(b.file).decode("utf-8"))

      bases = [Counter(bytes(b.base).decode("utf-8") for b in bbs0) for bbs0 in bbs]
      assert len(bbs) == len(queries)
      for i, v in enumerate(bbs):
         assert len(v) > 0, 'no bblocks for query: "' + queries[i] + '"'
      print("bblock queries:", str(queries))
      print("bblock numbers:", [len(b) for b in bbs])
      print("bblocks id:", [id(b) for b in bbs])
      print("bblock0 id ", [id(b[0]) for b in bbs])
      print("base_counts:")
      for query, basecount in zip(queries, bases):
         counts = " ".join(f"{k}: {c}" for k, c in basecount.items())
         print(f"   {query:10}", counts)

      if criteria.is_cyclic:
         # for a, b in zip(bbs[criteria.from_seg], bbs[criteria.to_seg]):
         # assert a is b
         bbs[criteria.to_seg] = bbs[criteria.from_seg]

      if use_saved_bblocks and not os.path.exists(savename):
         bbnames = [[bytes(b.file).decode("utf-8") for b in bb] for bb in bbs]
         with open(savename, "wb") as out:
            _pickle.dump(bbnames, out)

   else:
      bbs = bbs.copy()

   assert len(bbs) == len(criteria.bbspec)
   if modbbs:
      modbbs(bbs)

   if merge_bblock is not None and merge_bblock >= 0:
      # print('cloned_segments', criteria.bbspec, criteria.cloned_segments())
      if hasattr(criteria, "cloned_segments") and merge_segment is None:
         for i in criteria.cloned_segments():
            # print('   ', 'merge seg', i, 'merge_bblock', merge_bblock)
            bbs[i] = (bbs[i][merge_bblock], )
      else:
         if merge_segment is None:
            merge_segment = 0
         # print('   ', 'merge_segment not None')
         # print('   ', [len(b) for b in bbs])
         # print('   ', 'merge_segment', merge_segment)
         # print('   ', 'merge_bblock', merge_bblock, len(bbs[merge_segment]))
         bbs[merge_segment] = (bbs[merge_segment][merge_bblock], )

   tdb = time() - tdb
   # info(
   # f'bblock creation time {tdb:7.3f} num bbs: ' +
   # str([len(x) for x in bbs])
   # )

   if precache_splices:
      bbnames = [[bytes(bb.file) for bb in bbtup] for bbtup in bbs]
      bbpairs = set()
      # for bb1, bb2, dirn1 in zip(bbnames, bbnames[1:], directions):
      for i in range(len(bbnames) - 1):
         bb1 = bbnames[i]
         bb2 = bbnames[i + 1]
         dirn1 = directions[i]
         rev = dirn1[1] == "N"
         if bbs[i] is bbs[i + 1]:
            bbpairs.update((a, a) for a in bb1)
         else:
            bbpairs.update((b, a) if rev else (a, b) for a in bb1 for b in bb2)
      precompute_splicedb(db, bbpairs, verbosity=verbosity, parallel=parallel, **kw)
   if precache_only:
      return bbs

   verts = [None] * len(queries)
   edges = [None] * len(queries[1:])
   if source:
      srcdirn = [
          "".join("NC_"[d] for d in source.verts[i].dirn)
          for i in range(len(source.verts))
      ]  # yapf: disable
      srcverts, srcedges = list(), list()
      for i, bb in enumerate(bbs):
         for isrc, bbsrc in enumerate(source.bbs):

            # fragile code... detecting this way can be wrong
            # print(i, isrc, directions[i], srcdirn[isrc])
            if directions[i] != srcdirn[isrc]:
               continue
            if [b.filehash for b in bb] == [b.filehash for b in bbsrc]:
               # super hacky fix, really need to be passed info on what's what
               if srcverts and srcverts[-1] + 1 != isrc:
                  continue
               verts[i] = source.verts[isrc]
               srcverts.append(isrc)

      for i, bb in enumerate(zip(bbs, bbs[1:])):
         bb0, bb1 = bb
         for isrc, bbsrc in enumerate(zip(source.bbs, source.bbs[1:])):
            bbsrc0, bbsrc1 = bbsrc
            if directions[i] != srcdirn[isrc]:
               continue
            if directions[i + 1] != srcdirn[isrc + 1]:
               continue
            he = [b.filehash for b in bb0] == [b.filehash for b in bbsrc0]
            he &= [b.filehash for b in bb1] == [b.filehash for b in bbsrc1]
            if not he:
               continue
            edges[i] = source.edges[isrc]
            srcedges.append(isrc)

   if not make_edges:
      edges = []

   tvertex = time()
   exe = InProcessExecutor()

   if parallel:
      exe = cf.ThreadPoolExecutor(max_workers=parallel)
   with exe as pool:
      if only_seg is not None:
         save = bbs, directions
         bbs = [bbs[only_seg]]
         directions = [directions[only_seg]]
         verts = [verts[only_seg]]
      futures = list()
      for i, bb in enumerate(bbs):
         dirn = directions[i]
         if verts[i] is None:
            futures.append(pool.submit(Vertex, bb, dirn, min_seg_len=min_seg_len))
      verts_new = [f.result() for f in futures]
      isnone = [i for i in range(len(verts)) if verts[i] is None]
      for i, inone in enumerate(isnone):
         verts[inone] = verts_new[i]
         if source:
            print('use new vertex', inone)
      if only_ivertex:
         # raise NotImplementedError
         print("!!!!!!! using one ivertex !!!!!", only_ivertex, len(verts), [v.len for v in verts])
         if len(only_ivertex) != len(verts):
            print("NOT altering verts, len(only_ivertex)!=len(verts) continuing...",
                  "this is ok if part of a sub-protocol")
         else:
            for i, v in enumerate(verts):
               if v.len > 1:  # could already have been "trimmed"
                  assert only_ivertex[i] < v.len
                  v.reduce_to_only_one_inplace(only_ivertex[i])
               # print('x2exit', v.x2exit.shape)
               # print('x2orig', v.x2orig.shape)
               # print('ires', v.ires.shape)
               # print('isite', v.isite.shape)
               # print('ichain', v.ichain.shape)
               # print('ibblock', v.ibblock.shape)
               # print('inout', v.inout.shape, v.inout[10:])
               # print('inbreaks', v.inbreaks.shape, v.inbreaks[10:])
               # print('dirn', v.dirn.shape)
               # # assert 0
      # print(i, len(verts_new), len(verts))
      if isnone:
         assert i + 1 == len(verts_new)
      assert all(v for v in verts)
      if only_seg is not None:
         verts = [None] * only_seg + verts + [None] * (len(queries) - only_seg - 1)
         bbs, directions = save
   tvertex = time() - tvertex
   # info(
   # f'vertex creation time {tvertex:7.3f} num verts ' +
   # str([v.len if v else 0 for v in verts])
   # )

   if make_edges:
      tedge = time()
      for i, e in enumerate(edges):
         if e is not None:
            continue
         edges[i], edge_analysis = Edge(
            verts[i],
            bbs[i],
            verts[i + 1],
            bbs[i + 1],
            splicedb=spdb,
            verbosity=verbosity,
            precache_splices=precache_splices,
            **kw,
         )
         allok = all(x[6] for x in edge_analysis)
         if allok:
            continue
         print("=" * 80)
         print("info for edges with no valid splices", edges[i].total_allowed_splices())
         for tup in edge_analysis:
            iblk0, iblk1, ofst0, ofst1, ires0, ires1 = tup[:6]
            ok, f_clash, f_rms, f_ncontact, f_ncnh, f_nhc = tup[6:12]
            m_rms, m_ncontact, m_ncnh, m_nhc = tup[12:]
            if ok:
               continue
            assert len(bbs[i + 0]) > iblk0
            assert len(bbs[i + 1]) > iblk1
            print("=" * 80)
            print("egde Bblock A", bytes(bbs[i][iblk0].file))
            print("egde Bblock B", bytes(bbs[i + 1][iblk1].file))
            print(
               f"bb {iblk0:3} {iblk1:3}",
               f"ofst {ofst0:4} {ofst1:4}",
               f"resi {ires0.shape} {ires1.shape}",
            )
            print(
               f"clash_ok {int(f_clash*100):3}%",
               f"rms_ok {int(f_rms*100):3}%",
               f"ncontact_ok {int(f_ncontact*100):3}%",
               f"ncnh_ok {int(f_ncnh*100):3}%",
               f"nhc_ok {int(f_nhc*100):3}%",
            )
            print(
               f"min_rms {m_rms:7.3f}",
               f"max_ncontact {m_ncontact:7.3f}",
               f"max_ncnh {m_ncnh:7.3f}",
               f"max_nhc {m_nhc:7.3f}",
            )
         print("=" * 80)
         fok = np.stack([x[7:12] for x in edge_analysis]).mean(axis=0)
         rmsmin = np.array([x[12] for x in edge_analysis]).min()
         fmx = np.stack([x[13:] for x in edge_analysis]).max(axis=0)
         print(f"{' SPLICE FAIL SUMMARY ':=^80}")
         print(f"splice clash ok               {int(fok[0]*100):3}%")
         print(f"splice rms ok                 {int(fok[1]*100):3}%")
         print(f"splice ncontacts ok           {int(fok[2]*100):3}%")
         print(f"splice ncontacts_no_helix ok  {int(fok[3]*100):3}%")
         print(f"splice nhelixcontacted ok     {int(fok[4]*100):3}%")
         print(f"min rms of any failing        {rmsmin}")
         print(f"max ncontact of any failing   {fmx[0]} (maybe large for non-5-helix splice)")
         print(f"max ncontact_no_helix         {fmx[1]} (will be 999 for non-5-helix splice)")
         print(f"max nhelix_contacted          {fmx[2]} (will be 999 for non-5-helix splice)")
         print("=" * 80)
         assert edges[i].total_allowed_splices() > 0, "invalid splice"
      tedge = time() - tedge
      if print_edge_summary:
         _print_edge_summary(edges)
      # info(
      # f'edge creation time {tedge:7.3f} num splices ' +
      # str([e.total_allowed_splices()
      # for e in edges]) + ' num exits ' + str([e.len for e in edges])
      # )
      spdb.sync_to_disk()

   toret = SearchSpaceDag(criteria.bbspec, bbs, verts, edges)
   if timing:
      toret = toret, tdb, tvertex, tedge
   return toret


def _print_edge_summary(edges):
   print("splice stats: ", end="")
   for e in edges:
      nsplices = e.total_allowed_splices()
      ntot = e.nout * e.nent
      print(f"({nsplices:,} {nsplices*100.0/ntot:5.2f}%)", end=" ")
   print()


def graph_dump_pdb(out, ssdag, idx, pos, join="splice", trim=True):
   close = False
   if isinstance(out, str):
      out = open(out, "w")
      close = True
   assert len(idx) == len(pos)
   assert idx.ndim == 1
   assert pos.ndim == 3
   assert pos.shape[-2:] == (4, 4)
   chain, anum, rnum = 0, 1, 1
   for i, tup in enumerate(zip(ssdag.bbs, ssdag.verts, idx, pos)):
      bbs, vert, ivert, x = tup
      chain, anum, rnum = bblock_dump_pdb(
         out=out,
         bblock=bbs[vert.ibblock[ivert]],
         dirn=vert.dirn if trim else (2, 2),
         splice=vert.ires[ivert] if trim else (-1, -1),
         pos=x,
         chain=chain,
         anum=anum,
         rnum=rnum,
         join=join,
      )
   if close:
      out.close()
