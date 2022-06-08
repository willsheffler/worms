import string
from json import dumps
from deferred_import import deferred_import

import numpy as np
import numba
from scipy.spatial import ConvexHull

import worms
from worms.bblock.bbutil import make_connections_array, ncac_to_stubs
from worms.util import jitclass
from worms.util.util import generic_equals

def make_bblock(
   entry: dict,
   pdbfile: str,
   filehash: int,
   pose,
   ss,
   null_base_names,
   **kw,
):
   json = dumps(entry)
   chains = worms.util.rosetta_utils.get_chain_bounds(pose)
   ss = np.frombuffer(ss.encode(), dtype='i1')
   ncac = worms.util.rosetta_utils.get_bb_coords(pose)
   cb = worms.util.rosetta_utils.get_cb_coords(pose).astype(np.float32)
   stubs = ncac_to_stubs(ncac).astype(np.float32)
   com = np.mean(cb, axis=0).astype(np.float32)
   rg = np.sqrt(np.sum((cb - com)**2) / len(cb)).astype(np.float32)

   assert pose.size() == len(ncac)
   assert pose.size() == len(stubs)
   assert pose.size() == len(ss)
   conn = make_connections_array(entry['connections'], chains)
   if len(conn) == 0:
      print('bad conn info!', pdbfile)
      assert 0
      return None, pdbfile  # new, missing
   for c in conn:
      # is jagged array, padding is -1 so must be ignored
      assert np.all(c[:c[1]] >= 0), 'connection residues should all be positive at this point'

   if ncac.shape[-1] == 4:
      ncac = ncac.astype(np.float32)
   elif ncac.shape[-1] == 3:
      tmp = np.ones((ncac.shape[0], 3, 4), dtype=np.float32)
      tmp[..., :3] = ncac
      ncac = tmp
   else:
      assert 0, 'bad ncac'
   assert cb.shape == (pose.size(), 4)

   if entry['base'] in null_base_names: basehash = 0
   else: basehash = worms.util.hash_str_to_int(entry['base'])

   ca = ncac[:, 1, :]
   hullcoord = np.array([np.mean(ca[i - 3:i + 4], axis=0) for i in range(3, len(ca) - 4)])
   hullcoord = hullcoord.astype(np.float32)
   worms.PING(f'hullcoord shape {hullcoord.shape}')
   # assert 0

   from scipy.spatial.qhull import QhullError
   try:
      hull_obj = ConvexHull(hullcoord[:, :3])
      hull = hullcoord[hull_obj.vertices, :3]
   except (QhullError, IndexError):
      hull = np.empty((0, 3))

   numhull = len(hull)
   # print(hull.shape)
   # print(numhull)
   # assert 0

   validated = entry['validated']
   if validated in ('na', 'NA'):
      validated = False

   helixnum, helixresbeg, helixresend, helixbeg, helixend = worms.vertex.get_bb_helices(ss, ncac)

   repeataxis = np.array([0, 0, 0, 0], dtype=np.float32)

   bblock = _BBlock(
      json=npfb(json),
      connections=conn,
      file=npfb(entry['file']),
      filehash=filehash,
      components=npfb(str(entry['components'])),
      protocol=npfb(entry['protocol']),
      name=npfb(entry['name']),
      classes=npfb(','.join(entry['class'])),
      validated=validated,
      _type=npfb(entry['type']),
      base=npfb(entry['base']),
      basehash=basehash,
      ncac=np.ascontiguousarray(ncac),
      cb=np.ascontiguousarray(cb),
      chains=np.array(chains, dtype='i4'),
      ss=ss,
      stubs=np.ascontiguousarray(stubs),
      com=com,
      rg=rg,
      numhull=numhull,
      hull=hull,
      helixnum=helixnum,
      helixresbeg=helixresbeg,
      helixresend=helixresend,
      helixbeg=helixbeg,
      helixend=helixend,
      repeataxis=repeataxis,
   )

   return bblock


@jitclass(
    (
        ('json',        numba.types.int8[:]),
        ('connections', numba.types.int32[:, :]),
        ('file',        numba.types.int8[:]),
        ('filehash',    numba.types.int64),
        ('components',  numba.types.int8[:]),
        ('protocol',    numba.types.int8[:]),
        ('name',        numba.types.int8[:]),
        ('classes',     numba.types.int8[:]),
        ('validated',   numba.types.boolean),
        ('_type',       numba.types.int8[:]),
        ('base',        numba.types.int8[:]),
        ('basehash',    numba.types.int64),
        ('ncac',        numba.types.float32[:, :, :]),
        ('cb',          numba.types.float32[:, :]),
        ('chains',      numba.types.int32[:, :]),
        ('ss',          numba.types.int8[:]),
        ('stubs',       numba.types.float32[:, :, :]),
        ('com',         numba.types.float32[:]),
        ('rg',          numba.types.float32),
        ('numhull',     numba.types.int32),
        ('hull',        numba.types.float32[:,:] ),
        ('helixnum'   , numba.types.int32        ),
        ('helixresbeg', numba.types.int32[:]     ),
        ('helixresend', numba.types.int32[:]     ),
        ('helixbeg'   , numba.types.float32[:,:] ) ,
        ('helixend'   , numba.types.float32[:,:] ),
        ('repeataxis' , numba.types.float32[:] ),
    )
)  # yapf: disable
class _BBlock:
   '''member 'connections' is a jagged array. elements start at position 2. position 0 encodes the (N/C) direction as 0, 1, or 2, decoded as 'NC_', position 1 is the ending location of residue entries (so the number of residues is 2 less, leaving off the first two entries) see the member functions
'''
   def __init__(self, json, connections, file, filehash, components, protocol, name, classes,
                validated, _type, base, basehash, ncac, cb, chains, ss, stubs, com, rg, numhull,
                hull, helixnum, helixresbeg, helixresend, helixbeg, helixend, repeataxis):
      self.json = json
      self.connections = connections
      self.file = file
      self.filehash = filehash
      self.components = components
      self.protocol = protocol
      self.name = name
      self.classes = classes
      self.validated = validated
      self._type = _type
      self.base = base
      self.basehash = basehash
      self.ncac = ncac
      self.cb = cb
      self.chains = chains
      self.ss = ss
      self.stubs = stubs
      self.com = com
      self.rg = rg

      self.numhull = numhull
      self.hull = hull.astype(np.float32)

      self.helixnum = helixnum
      self.helixresbeg = helixresbeg
      self.helixresend = helixresend
      self.helixbeg = helixbeg
      self.helixend = helixend

      self.repeataxis = repeataxis

      assert np.isnan(np.sum(self.ncac)) == False
      assert np.isnan(np.sum(self.cb)) == False
      assert np.isnan(np.sum(self.stubs)) == False
      assert np.isnan(np.sum(self.ss)) == False
      assert np.isnan(np.sum(self.chains)) == False

   @property
   def n_connections(self):
      return len(self.connections)

   def conn_dirn(self, i):
      return self.connections[i, 0]

   def conn_resids(self, i):
      return self.connections[i, 2:self.connections[i, 1]]

   @property
   def _state(self):
      # MUST stay same order as args to __init__!!!!!
      return (self.json, self.connections, self.file, self.filehash, self.components,
              self.protocol, self.name, self.classes, self.validated, self._type, self.base,
              self.basehash, self.ncac, self.cb, self.chains, self.ss, self.stubs, self.com,
              self.rg, self.numhull, self.hull, self.helixnum, self.helixresbeg, self.helixresend,
              self.helixbeg, self.helixend, self.repeataxis)

   def __setstate__(self, state):
      (self.json, self.connections, self.file, self.filehash, self.components, self.protocol,
       self.name, self.classes, self.validated, self._type, self.base, self.basehash, self.ncac,
       self.cb, self.chains, self.ss, self.stubs, self.com, self.rg, self.numhull, self.hull,
       self.helixnum, self.helixresbeg, self.helixresend, self.helixbeg, self.helixend,
       self.repeataxis) = state

   def __getstate__(self):
      return self._state

   def equal_to(self, other):
      # return generic_equals(self._state, other._state)
      # dunno if it's safe to ignore ncac, chains and stubs
      # but they make the comparison slow
      with numba.objmode(eq='b1'):
         eq = all([
            generic_equals(self.json, other.json),
            generic_equals(self.connections, other.connections),
            generic_equals(self.file, other.file),
            generic_equals(self.filehash, other.filehash),
            generic_equals(self.components, other.components),
            generic_equals(self.protocol, other.protocol),
            generic_equals(self.name, other.name),
            generic_equals(self.classes, other.classes),
            generic_equals(self.validated, other.validated),
            generic_equals(self._type, other._type),
            generic_equals(self.base, other.base),
            generic_equals(self.basehash, other.basehash),
            # generic_equals(self.ncac, other.ncac),
            generic_equals(self.cb, other.cb),
            generic_equals(self.chains, other.chains),
            generic_equals(self.ss, other.ss),
            # generic_equals(self.stubs, other.stubs),
            generic_equals(self.com, other.com),
            generic_equals(self.rg, other.rg),
            generic_equals(self.numhull, other.numhull),
            generic_equals(self.hull, other.hull),
            generic_equals(self.helixnum, self.helixnum),
            generic_equals(self.helixresbeg, self.helixresbeg),
            generic_equals(self.helixresend, self.helixresend),
            generic_equals(self.helixbeg, self.helixbeg),
            generic_equals(self.helixend, self.helixend),
         ])
      return eq

class BBlock:
   def __init__(self, _bblock):
      self._bblock = _bblock

   @property
   def ss(self):
      # return np.array(self._bblock.ss)
      return self._bblock.ss

   @property
   def ncac(self):
      # return np.array(self._bblock.ncac)
      return self._bblock.ncac

   @property
   def conn_directions(self):
      n = self._bblock.n_connections
      return [self._bblock.conn_dirn(i) for i in range(n)]

   @property
   def conn_residues(self):
      n = self._bblock.n_connections
      return [self._bblock.conn_resids(i) for i in range(n)]

   @property
   def connections(self):
      return list(zip(self.conn_directions, self.conn_residues))

   def helixinfo(self, reslb=0, resub=99999, trim=10):
      hrb, hre = self._bblock.helixresbeg, self._bblock.helixresend
      ok = np.logical_and(hrb <= resub + trim, hre >= reslb - trim)
      return (
         sum(ok),
         hrb[ok],
         hre[ok],
         self._bblock.helixbeg[ok],
         self._bblock.helixend[ok],
      )

   @property
   def classes(self):
      return unnpfb(self._bblock.classes)

   @property
   def is_cyclic(self):
      c = self.classes
      if len(c) < 4: return False
      return all([
         c[0] == 'C',
         c[1].isdigit(),
         c[2] == '_',
         c[3] in 'CN',
      ])

   @property
   def chains(self):
      return np.array(self._bblock.chains)

   def __setstate__(self, state):
      self._bblock = _BBlock(*state)

   def __getstate__(self):
      return self._bblock._state

   @property
   def com(self):
      return self._bblock.com

   @property
   def json(self):
      return bytes(self._bblock.json).decode()
      # self.components = components
      # self.protocol = protocol
      # self.name = name
      # self.classes = classes
      # self.validated = validated
      # self._type = _type
      # self.base = base

def npfb(s):
   if isinstance(s, list):
      s = '[' + ','.join(s) + ']'
   return np.frombuffer(s.encode(), dtype='i1')

def unnpfb(fb):
   return bytes(fb).decode()
