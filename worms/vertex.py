"""TODO: Summary
"""

import numpy as np
import numba as nb
import numba.types as nt
from homog import is_homog_xform
from worms import util
from worms.bblock import chain_of_ires, _BBlock
from logging import warning
import concurrent.futures as cf
from worms.util import InProcessExecutor, jit
from worms.criteria import cyclic

vertex_xform_dtype = np.float32


@nb.jitclass(
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
    )
)  # yapf: disable
class _Vertex:
   """contains data for one topological vertex in the topological ssdag

    Attributes:
        dirn (TYPE): Description
        ibblock (TYPE): Description
        ichain (TYPE): Description
        inout (TYPE): Description
        ires (TYPE): Description
        isite (TYPE): Description
        x2exit (TYPE): Description
        x2orig (TYPE): Description
    """

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
   ):
      """TODO: Summary

        Args:
            x2exit (TYPE): Description
            x2orig (TYPE): Description
            ires (TYPE): Description
            isite (TYPE): Description
            ichain (TYPE): Description
            ibblock (TYPE): Description
            inout (TYPE): Description
            dirn (TYPE): Description

        Deleted Parameters:
            bblock (TYPE): Description
        """
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


@jit
def _check_inorder(ires):
   for i in range(len(ires) - 1):
      if ires[i] > ires[i + 1]:
         return False
   return True


def vertex_single(bbstate, bbid, din, dout, min_seg_len, verbosity=0):
   """build on bblock's worth of vertex"""
   bb = _BBlock(*bbstate)
   ires0, ires1 = [], []
   isite0, isite1 = [], []
   for i in range(bb.n_connections):
      ires = bb.conn_resids(i)
      if bb.conn_dirn(i) == din:
         ires0.append(ires)
         isite0.append(np.repeat(i, len(ires)))
      if bb.conn_dirn(i) == dout:
         ires1.append(ires)
         isite1.append(np.repeat(i, len(ires)))
   dirn = "NC_" [din] + "NC_" [dout]
   if din < 2 and not ires0 or dout < 2 and not ires1:
      if verbosity > 0:
         warning("invalid vertex " + dirn + " " + bytes(bb.file).decode())
      return None

   dummy = [np.array([-1], dtype="i4")]
   ires0 = np.concatenate(ires0 or dummy)
   ires1 = np.concatenate(ires1 or dummy)
   isite0 = np.concatenate(isite0 or dummy)
   isite1 = np.concatenate(isite1 or dummy)
   chain0 = chain_of_ires(bb, ires0)
   chain1 = chain_of_ires(bb, ires1)

   if ires0[0] == -1:
      assert len(ires0) is 1
   else:
      assert np.all(ires0 >= 0)
   if ires1[0] == -1:
      assert len(ires1) is 1
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

   assert _check_inorder(ires0)
   assert _check_inorder(ires1)

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


def Vertex(bbs, dirn, bbids=None, min_seg_len=1, verbosity=0):
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
       [
           util.unique_key_int32s(ibblock, ires[:, 0]),
           util.unique_key_int32s(ibblock, ires[:, 1]),
       ],
       axis=-1,
   ).astype(
       "i4"
   )  # yapf: disable

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

   inbreaks = util.contig_idx_breaks(inout[:, 0])
   assert inbreaks.dtype == np.int32
   assert np.all(inbreaks <= len(inout))

   return _Vertex(*tup, inout, inbreaks, np.array([din, dout], dtype="i4"), min_seg_len)
