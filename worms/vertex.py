"""TODO: Summary
"""

from logging import warning
import concurrent.futures as cf

import numpy as np
import numba as nb
import numba.types as nt

import worms
from worms import util
from worms.homog import is_homog_xform
from worms.bblock import _BBlock
from worms.util import InProcessExecutor, jit, jitclass, helix_range
from worms.criteria import cyclic
from worms.util.util import generic_equals

vertex_xform_dtype = np.float32

MAX_HELIX = 128
MAX_HULL = 512

@jitclass(
      (
            ("x2exit", nb.typeof(vertex_xform_dtype(0))[:, :, :]),
            ("x2orig", nb.typeof(vertex_xform_dtype(0))[:, :, :]),
            ("inout", nt.int32[:, :]),
            ("inbreaks", nt.int32[:]),
            ("ires", nt.int32[:, :]),
            ("isite", nt.int32[:, :]),
            ("ichain", nt.int32[:, :]),
            ("ibblock", nt.int32[:]),
            ("dirn", nt.int32[:]),
            ("min_seg_len", nt.int32),
            ('numhelix', nt.int32[:]),
            ('helixresbeg', nt.int32[:,:]),
            ('helixresend', nt.int32[:,:]),
            ('helixbeg', nt.float32[:,:,:]),
            ('helixend', nt.float32[:,:,:]),
            ('numhull', nt.int32[:]),
            ('hull', nt.float32[:,:,:])
      )
)  # yapf: disable
class _Vertex:
   def __init__(
      self,
      x2exit,
      x2orig,
      ires,
      isite,
      ichain,
      ibblock,
      inout,
      inbreaks,
      dirn,
      min_seg_len,
      numhelix,
      helixresbeg,
      helixresend,
      helixbeg,
      helixend,
      numhull,
      hull,
   ):
      self.x2exit = x2exit.astype(vertex_xform_dtype)
      self.x2orig = x2orig.astype(vertex_xform_dtype)
      self.ires = ires
      self.isite = isite
      self.ichain = ichain
      self.ibblock = ibblock
      self.inout = inout
      self.inbreaks = inbreaks
      self.dirn = dirn
      self.min_seg_len = min_seg_len
      self.numhelix = numhelix
      self.helixresbeg = helixresbeg
      self.helixresend = helixresend
      self.helixbeg = helixbeg
      self.helixend = helixend

      self.numhull = numhull
      self.hull = hull

   @property
   def entry_index(self):
      return self.inout[:, 0]

   @property
   def exit_index(self):
      return self.inout[:, 1]

   def entry_range(self, ienter):
      assert ienter >= 0, "vertex.py bad ienter, < 0"
      assert ienter <= len(self.inbreaks), "vertex.py bad ienter"
      return self.inbreaks[ienter], self.inbreaks[ienter + 1]

   def reduce_to_only_one_inplace(self, idx):
      self.x2exit = self.x2exit[idx:idx + 1]
      self.x2orig = self.x2orig[idx:idx + 1]
      self.ires = self.ires[idx:idx + 1]
      self.isite = self.isite[idx:idx + 1]
      self.ichain = self.ichain[idx:idx + 1]
      self.ibblock = self.ibblock[idx:idx + 1]
      self.inout = np.zeros((1, 2), dtype=np.int32)
      self.inbreaks = np.zeros(2, dtype=np.int32)
      self.inbreaks[1] = 1

   @property
   def len(self):
      return len(self.ires)

   @property
   def _state(self):
      return (
         self.x2exit,
         self.x2orig,
         self.ires,
         self.isite,
         self.ichain,
         self.ibblock,
         self.inout,
         self.inbreaks,
         self.dirn,
         self.min_seg_len,
         self.numhelix,
         self.helixresbeg,
         self.helixresend,
         self.helixbeg,
         self.helixend,
         self.numhull,
         self.hull,
      )

   @property
   def memuse(self):
      return (self.x2exit.size * self.x2exit.itemsize + self.x2orig.size * self.x2orig.itemsize)
      # ('inout'   , nt.int32[:, :]),
      # ('inbreaks', nt.int32[:]),
      # ('ires'    , nt.int32[:, :]),
      # ('isite'   , nt.int32[:, :]),
      # ('ichain'  , nt.int32[:, :]),
      # ('ibblock' , nt.int32[:]),
      # ('dirn'    , nt.int32[:]),
      # ('min_seg_len', nt.int32),

   def equal_to(self, other):
      with nb.objmode(eq='b1'):
         eq = generic_equals(self._state, other._state)
      return eq

@jit
def _check_inorder_nodups(ires):
   for i in range(len(ires) - 1):
      if ires[i] > ires[i + 1]:
         return False
   return True

def vertex_single(bbstate, bbid, din, dout, min_seg_len, verbosity=0):
   """build on bblock's worth of vertex"""
   bb = _BBlock(*bbstate)
   ires0, ires1 = [], []
   isite0, isite1 = [], []
   for isite in range(bb.n_connections):
      ires = bb.conn_resids(isite)
      if bb.conn_dirn(isite) == din:
         ires0.append(ires)
         isite0.append(np.repeat(isite, len(ires)))
      if bb.conn_dirn(isite) == dout:
         ires1.append(ires)
         isite1.append(np.repeat(isite, len(ires)))
   dirn = "NC_"[din] + "NC_"[dout]
   if din < 2 and not ires0 or dout < 2 and not ires1:
      if verbosity > 0:
         warning("invalid vertex " + dirn + " " + bytes(bb.file).decode())
      return None

   nres = len(bb.ncac)
   # assert len(bb.ncac) == len(bb.stubs)
   # print('nres', nres, len(bb.stubs))
   # print('ires0')
   # print(ires0)
   # print('ires1')
   # print(ires1)
   # print()

   # for ary in ires0:
   #    for i, ir in enumerate(ary):
   #       if ir < 0:
   #          ary[i] = nres + ir
   #    assert np.all(ary >= 0)
   #    assert np.all(ary < nres)
   # for ary in ires1:
   #    for i, ir in enumerate(ary):
   #       if ir < 0:
   #          ary[i] = nres + ir
   #    assert np.all(ary >= 0)
   #    assert np.all(ary < nres)

   dummy = [np.array([-1], dtype="i4")]
   ires0 = np.concatenate(ires0 or dummy)
   ires1 = np.concatenate(ires1 or dummy)
   isite0 = np.concatenate(isite0 or dummy)
   isite1 = np.concatenate(isite1 or dummy)
   chain0 = chain_of_ires(bb, ires0)
   chain1 = chain_of_ires(bb, ires1)

   if ires0[0] == -1:
      assert len(ires0) == 1
   else:

      assert np.all(ires0 >= 0)
   if ires1[0] == -1:
      assert len(ires1) == 1
   else:
      assert np.all(ires1 >= 0)

   if ires0[0] == -1:
      stub0inv = np.eye(4).reshape(1, 4, 4)
   else:
      stub0inv = np.linalg.inv(bb.stubs[ires0])
   if ires1[0] == -1:
      stub1 = np.eye(4).reshape(1, 4, 4)
   else:
      stub1 = bb.stubs[ires1]

   assert _check_inorder_nodups(ires0), ires0
   assert _check_inorder_nodups(ires1), ires1

   # print(ires1)
   # import sys
   # sys.exit()

   stub0inv, stub1 = np.broadcast_arrays(stub0inv[:, None], stub1)
   ires = np.stack(np.broadcast_arrays(ires0[:, None], ires1), axis=-1)
   isite = np.stack(np.broadcast_arrays(isite0[:, None], isite1), axis=-1)
   chain = np.stack(np.broadcast_arrays(chain0[:, None], chain1), axis=-1)

   x2exit = stub0inv @ stub1
   x2orig = stub0inv
   # assert is_homog_xform(x2exit)  # this could be slowish
   # assert is_homog_xform(x2orig)

   # min chain len, not same site
   not_same_chain = chain[..., 0] != chain[..., 1]
   not_same_site = isite[..., 0] != isite[..., 1]
   seqsep = np.abs(ires[..., 0] - ires[..., 1])

   # remove invalid in/out pairs (+ is or, * is and)
   valid = not_same_site
   valid *= not_same_chain + (seqsep >= min_seg_len)
   valid = valid.reshape(-1)

   if np.sum(valid) == 0:
      return None

   return (
      x2exit.reshape(-1, 4, 4)[valid],
      x2orig.reshape(-1, 4, 4)[valid],
      ires.reshape(-1, 2)[valid].astype("i4"),
      isite.reshape(-1, 2)[valid].astype("i4"),
      chain.reshape(-1, 2)[valid].astype("i4"),
      np.repeat(bbid, np.sum(valid)).astype("i4"),
   )

@jit
def _check_bbires_inorder(ibblock, ires):
   prev = -np.ones(np.max(ibblock) + 1, dtype=np.int32)
   for i in range(len(ires)):
      if ires[i] >= 0:
         if ires[i] < prev[ibblock[i]]:
            # print('_check_bbires_inorder err', i)
            return False
         prev[ibblock[i]] = ires[i]
   return True

@jit
def chain_of_ires(bb, ires):
   chain = np.empty_like(ires)
   for i, ir in enumerate(ires):
      if ir < 0:
         chain[i] = -1
      else:
         for c in range(len(bb.chains)):
            if bb.chains[c, 0] <= ir < bb.chains[c, 1]:
               chain[i] = c
   return chain

def get_bb_helices(ss, ncac, trim=0, **kw):

   minsize = kw['helixconf_min_helix_size'] if 'helixconf_min_helix_size' in kw else 14
   maxsize = kw['helixconf_max_helix_size'] if 'helixconf_max_helix_size' in kw else 999
   helixresbeg, helixresend, helixbeg, helixend = list(), list(), list(), list()
   hrange, helixof = helix_range(ss)
   for ih, (lb, ub) in enumerate(hrange):
      if ub - lb < minsize: continue
      if ub - lb > maxsize: continue
      beg = np.mean(ncac[(lb + trim):(lb + trim + 7), 1], axis=0)
      end = np.mean(ncac[(ub - trim - 6):(ub - trim + 1), 1], axis=0)
      # beg = np.mean(ncac[lb + 3:lb + 10, 1], axis=0)
      # end = np.mean(ncac[ub - 9:ub - 2, 1], axis=0)
      helixresbeg.append(lb)
      helixresend.append(ub)
      helixbeg.append(beg)
      helixend.append(end)
   numh = len(helixend)

   helixresbeg = np.array(helixresbeg, dtype=np.int64)
   helixresend = np.array(helixresend, dtype=np.int64)
   helixbeg = np.array(helixbeg, dtype=np.float64)
   helixend = np.array(helixend, dtype=np.float64)
   if numh == 0:
      helixresbeg = np.array([], dtype=np.int64)
      helixresend = np.array([], dtype=np.int64)
      helixbeg = np.array([[]], dtype=np.float64)
      helixend = np.array([[]], dtype=np.float64)

   assert isinstance(helixbeg, np.ndarray) and helixbeg.ndim == 2
   assert isinstance(helixend, np.ndarray) and helixend.ndim == 2

   return numh, helixresbeg, helixresend, helixbeg, helixend

def Vertex(bbs, dirn, bbids=None, min_seg_len=1, verbosity=0, **kw):

   from worms.util.jitutil import unique_key_int32s, contig_idx_breaks

   dirn_map = {"N": 0, "C": 1, "_": 2}
   din = dirn_map[dirn[0]]
   dout = dirn_map[dirn[1]]
   if bbids is None:
      bbids = np.arange(len(bbs))

   # exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
   # with exe() as pool:
   #     futures = list()
   #     for bb, bid in zip(bbs, bbids):
   #         futures.append(
   #             pool.
   #             submit(vertex_single, bb._state, bid, din, dout, min_seg_len)
   #         )
   #     verts = [f.result() for f in futures]
   verts = [
      vertex_single(bb._state, bid, din, dout, min_seg_len, verbosity=verbosity)
      for bb, bid in zip(bbs, bbids)
   ]
   verts = [v for v in verts if v is not None]

   if not verts:
      raise ValueError("no way to make vertex: '" + dirn + "'")
   tup = tuple(np.concatenate(_) for _ in zip(*verts))
   assert len({x.shape[0] for x in tup}) == 1
   ibblock, ires = tup[5], tup[2]

   # print(np.stack((ibblock, ires[:, 1])).T)

   assert _check_bbires_inorder(ibblock, ires[:, 0])
   # not true as some pruned from validity checks
   # assert _check_bbires_inorder(ibblock, ires[:, 1])

   inout = np.stack(
      [unique_key_int32s(ibblock, ires[:, 0]),
       unique_key_int32s(ibblock, ires[:, 1])],
      axis=-1,
   ).astype("i4")

   # inout2 = np.stack([
   #     util.unique_key(ibblock, ires[:, 0]),
   #     util.unique_key(ibblock, ires[:, 1])
   # ],
   #                  axis=-1).astype('i4')
   # if not np.all(inout == inout2):
   #     np.set_printoptions(threshold=np.nan)
   #     print(
   #         np.stack((
   #             inout[:, 0], inout2[:, 0], ibblock, ires[:, 0], inout[:, 1],
   #             inout2[:, 1], ibblock, ires[:, 1]
   #         )).T
   #     )

   # assert inout.shape == inout2.shape
   # assert np.all(inout == inout2)

   inbreaks = contig_idx_breaks(inout[:, 0])
   assert inbreaks.dtype == np.int32
   assert np.all(inbreaks <= len(inout))

   numhelix = np.zeros(dtype=np.int32, shape=(len(bbs), ))
   helixresbeg = np.zeros(dtype=np.int32, shape=(len(bbs), MAX_HELIX))
   helixresend = np.zeros(dtype=np.int32, shape=(len(bbs), MAX_HELIX))
   helixbeg = np.zeros(dtype=np.float32, shape=(len(bbs), MAX_HELIX, 4))
   helixend = np.zeros(dtype=np.float32, shape=(len(bbs), MAX_HELIX, 4))

   for ibb, bb in enumerate(bbs):

      numh, helixresbeg0, helixresend0, helixbeg0, helixend0 = get_bb_helices(bb.ss, bb.ncac)

      numhelix[ibb] = numh
      if numh:
         helixresbeg[ibb, :numh] = helixresbeg0
         helixresend[ibb, :numh] = helixresend0
         helixbeg[ibb, :numh] = helixbeg0
         helixend[ibb, :numh] = helixend0

      numhull = -np.ones(dtype=np.int32, shape=(len(bbs), ))
      hull = 9e9 * np.ones(dtype=np.float32, shape=(len(bbs), MAX_HULL, 4))
      hull[:, :, 3] = 1
      numhull[ibb] = bb.numhull
      hull[ibb, :bb.numhull, :3] = bb.hull
      # print(bb.hull.shape)
      # print(hull[ibb, :bb.numhull, :].shape)
      # print(hull.shape)
      # assert 0

   # print('numhelix', numhelix)
   # print()
   # print(helixbeg[0, :10])
   # print()
   # print(helixbeg[1, :10])
   # print('helixori', numhelix.shape, helixbeg.shape, helixend.shape)
   # assert 0

   return _Vertex(
      *tup,
      inout,
      inbreaks,
      np.array([din, dout], dtype="i4"),
      min_seg_len,
      numhelix,
      helixresbeg,
      helixresend,
      helixbeg,
      helixend,
      numhull,
      hull,
   )
