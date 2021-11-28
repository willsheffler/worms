import numpy as np
from worms import homog as hm

def make_connections_array(entries, chain_bounds):
   # try:
   # if True:
   reslists = [get_connection_residues(e, chain_bounds) for e in entries]
   # except Exception as e:
   # print("make_connections_array failed on", entries, "error was:", e)
   # return np.zeros((0, 0))

   order = np.argsort([x[0] for x in reslists])
   mx = max(len(x) for x in reslists)
   conn = np.zeros((len(reslists), mx + 2), "i4") - 1
   for i, iord in enumerate(order):
      conn[i, 0] = entries[iord]["direction"] == "C"
      conn[i, 1] = len(reslists[iord]) + 2
      conn[i, 2:conn[i, 1]] = reslists[iord]
   return conn

def get_connection_residues(entry, chain_bounds):
   """should return sorted list of resi positions"""
   # chain_bounds[-1][-1] # TODO removed
   r, c, d = entry["residues"], int(entry["chain"]), entry["direction"]
   nres = chain_bounds[c - 1][1] - chain_bounds[c - 1][0]
   if isinstance(r, str) and r.startswith("["):
      r = eval(r)
   if isinstance(r, list):
      try:
         return sorted(int(i if i >= 0 else i + nres) for i in r)
      except (TypeError, ValueError):
         assert len(r) == 1
         r = r[0]
   if r.count(","):
      c2, r = r.split(",")
      assert int(c2) == c
   b, e = r.split(":")
   if b == "-":
      b = 0
   if e == "-":
      e = -1
   nres = chain_bounds[c - 1][1] - chain_bounds[c - 1][0]
   b = int(b) if b else 0
   e = int(e) if e else nres
   if e < 0:
      e += nres
   return np.array(range(*chain_bounds[c - 1])[b:e], dtype="i4")

def bblock_components(bblock):
   return eval(bytes(bblock.components))

def bblock_str(bblock):
   return "\n".join([
      "jitclass BBlock(",
      "    file=" + str(bytes(bblock.file)),
      "    components=" + str(bblock_components(bblock)),
      "    protocol=" + str(bytes(bblock.protocol)),
      "    name=" + str(bytes(bblock.name)),
      "    classes=" + str(bytes(bblock.classes)),
      "    validated=" + str(bblock.validated),
      "    _type=" + str(bytes(bblock._type)),
      "    base=" + str(bytes(bblock.base)),
      "    ncac=array(shape=" + str(bblock.ncac.shape) + ", dtype=" + str(bblock.ncac.dtype) +
      ")",
      "    chains=" + str(bblock.chains),
      "    ss=array(shape=" + str(bblock.ss.shape) + ", dtype=" + str(bblock.ss.dtype) + ")",
      "    stubs=array(shape=" + str(bblock.stubs.shape) + ", dtype=" +
      str(bblock.connections.dtype) + ")",
      "    connectionsZ=array(shape=" + str(bblock.connections.shape) + ", dtype=" +
      str(bblock.connections.dtype) + ")",
      ")",
   ])

def bb_splice_res(bb, dirn):
   r = []
   for iconn in range(bb.n_connections):
      if bb.conn_dirn(iconn) == dirn:
         r.append(bb.conn_resids(iconn))
   return np.concatenate(r)

def bb_splice_res_N(bb):
   return splice_res(bb, 0)

def bb_splice_res_C(bb):
   return splice_res(bb, 1)

def ncac_to_stubs(ncac):
   """
        Vector const & center,
        Vector const & a,
        Vector const & b,
        Vector const & c
    )
    {
        Vector e1( a - b);
        e1.normalize();

        Vector e3( cross( e1, c - b ) );
        e3.normalize();

        Vector e2( cross( e3,e1) );
        M.col_x( e1 ).col_y( e2 ).col_z( e3 );
        v = center;
    """
   assert ncac.shape[1:] == (3, 4)
   stubs = np.zeros((len(ncac), 4, 4), dtype=np.float64)
   ca2n = (ncac[:, 0] - ncac[:, 1])[..., :3]
   ca2c = (ncac[:, 2] - ncac[:, 1])[..., :3]
   # tgt1 = ca2n + ca2c  # thought this might make
   # tgt2 = ca2n - ca2c  # n/c coords match better
   tgt1 = ca2n  # rosetta style
   tgt2 = ca2c  # seems better
   a = tgt1
   a /= np.linalg.norm(a, axis=-1)[:, None]
   c = np.cross(a, tgt2)
   c /= np.linalg.norm(c, axis=-1)[:, None]
   b = np.cross(c, a)
   assert np.allclose(np.sum(a * b, axis=-1), 0)
   assert np.allclose(np.sum(b * c, axis=-1), 0)
   assert np.allclose(np.sum(c * a, axis=-1), 0)
   assert np.allclose(np.linalg.norm(a, axis=-1), 1)
   assert np.allclose(np.linalg.norm(b, axis=-1), 1)
   assert np.allclose(np.linalg.norm(c, axis=-1), 1)
   stubs[:, :3, 0] = a
   stubs[:, :3, 1] = b
   stubs[:, :3, 2] = c
   stubs[:, :3, 3] = ncac[:, 1, :3]
   stubs[:, 3, 3] = 1
   return stubs
