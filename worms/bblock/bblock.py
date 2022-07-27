import enum, math, statistics, copy, string, json
from difflib import SequenceMatcher
from json import dumps
from deferred_import import deferred_import

import numpy as np
import numba
from scipy.spatial import ConvexHull

import worms
from worms.bblock.bbutil import make_connections_array, ncac_to_stubs
from worms.util import jitclass
from worms.util.util import generic_equals, add_props_to_url

import willutil as wu

def make_bblock(
   entry: dict,
   pose,
   null_base_names,
   **kw,
):
   pdbfile = entry['file']
   filehash = worms.util.hash_str_to_int(pdbfile)
   ss = worms.rosetta_init.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
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

   _bblock = _BBlock(
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
      sequence=npfb(pose.sequence()),
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
      repeataxis=np.array([0, 0, 0, 0], dtype=np.float32),
      repeatstart=-1,
      repeatspacing=-1,
   )

   assert np.all(_bblock.repeataxis == [0, 0, 0, 0])
   assert _bblock.repeatstart == -1
   assert _bblock.repeatspacing == -1
   if any(c.lower().startswith('straight') for c in entry['class']):
      print('-' * 80)
      print('get_repeat_axis', entry['file'])
      print('get_repeat_axis', entry['class'])
      bblock = BBlock(_bblock)

      start, spacing = get_repeat_spacing(bblock, **kw)
      print('bblock.py repeat start, spacing', start, spacing)

      if start is None:
         pose.dump_pdb('make_bblock.pdb')
         assert 0

      _bblock.repeatstart = start
      _bblock.repeatspacing = spacing
      repeataxis, repeataxiserr = get_repeat_axis(bblock, start, spacing, **kw)
      # print(repeataxis)
      # print(repeataxiserr)
      assert repeataxiserr < 1.0
      # assert 0
      _bblock.repeataxis = repeataxis

   return _bblock


@jitclass(
    (
        ('json',          numba.types.int8[:]),
        ('connections',   numba.types.int32[:, :]),
        ('file',          numba.types.int8[:]),
        ('filehash',      numba.types.int64),
        ('components',    numba.types.int8[:]),
        ('protocol',      numba.types.int8[:]),
        ('name',          numba.types.int8[:]),
        ('classes',       numba.types.int8[:]),
        ('validated',     numba.types.boolean),
        ('_type',         numba.types.int8[:]),
        ('base',          numba.types.int8[:]),
        ('basehash',      numba.types.int64),
        ('ncac',          numba.types.float32[:, :, :]),
        ('cb',            numba.types.float32[:, :]),
        ('chains',        numba.types.int32[:, :]),
        ('sequence',      numba.types.int8[:]),
        ('ss',            numba.types.int8[:]),
        ('stubs',         numba.types.float32[:, :, :]),
        ('com',           numba.types.float32[:]),
        ('rg',            numba.types.float32),
        ('numhull',       numba.types.int32),
        ('hull',          numba.types.float32[:,:] ),
        ('helixnum'   ,   numba.types.int32        ),
        ('helixresbeg',   numba.types.int32[:]     ),
        ('helixresend',   numba.types.int32[:]     ),
        ('helixbeg'   ,   numba.types.float32[:,:] ),
        ('helixend'   ,   numba.types.float32[:,:] ),
        ('repeataxis' ,   numba.types.float32[:]   ),
        ('repeatstart'  , numba.types.int32      ),
        ('repeatspacing', numba.types.int32      ),
    )
)  # yapf: disable
class _BBlock:
   '''member 'connections' is a jagged array. elements start at position 2. position 0 encodes the (N/C) direction as 0, 1, or 2, decoded as 'NC_', position 1 is the ending location of residue entries (so the number of residues is 2 less, leaving off the first two entries) see the member functions
'''
   def __init__(
      self,
      json,
      connections,
      file,
      filehash,
      components,
      protocol,
      name,
      classes,
      validated,
      _type,
      base,
      basehash,
      ncac,
      cb,
      chains,
      sequence,
      ss,
      stubs,
      com,
      rg,
      numhull,
      hull,
      helixnum,
      helixresbeg,
      helixresend,
      helixbeg,
      helixend,
      repeataxis,
      repeatstart,
      repeatspacing,
   ):
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
      self.sequence = sequence
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
      self.repeatstart = repeatstart
      self.repeatspacing = repeatspacing

      assert np.isnan(np.sum(self.ncac)) == False
      assert np.isnan(np.sum(self.cb)) == False
      assert np.isnan(np.sum(self.stubs)) == False
      assert np.isnan(np.sum(self.sequence)) == False
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
      return (
         self.json,
         self.connections,
         self.file,
         self.filehash,
         self.components,
         self.protocol,
         self.name,
         self.classes,
         self.validated,
         self._type,
         self.base,
         self.basehash,
         self.ncac,
         self.cb,
         self.chains,
         self.sequence,
         self.ss,
         self.stubs,
         self.com,
         self.rg,
         self.numhull,
         self.hull,
         self.helixnum,
         self.helixresbeg,
         self.helixresend,
         self.helixbeg,
         self.helixend,
         self.repeataxis,
         self.repeatstart,
         self.repeatspacing,
      )

   def __setstate__(self, state):
      (
         self.json,
         self.connections,
         self.file,
         self.filehash,
         self.components,
         self.protocol,
         self.name,
         self.classes,
         self.validated,
         self._type,
         self.base,
         self.basehash,
         self.ncac,
         self.cb,
         self.chains,
         self.sequence,
         self.ss,
         self.stubs,
         self.com,
         self.rg,
         self.numhull,
         self.hull,
         self.helixnum,
         self.helixresbeg,
         self.helixresend,
         self.helixbeg,
         self.helixend,
         self.repeataxis,
         self.repeatstart,
         self.repeatspacing,
      ) = state

   def __getstate__(self):
      return self._state

   def get_pdb_file(self):
      return unnpfb(self.file)

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
            generic_equals(self.sequence, other.sequence),
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
   def __init__(self, _bblock: _BBlock):
      assert isinstance(_bblock, _BBlock)
      self._bblock = _bblock

   @property
   def sequence(self):
      # return np.array(self._bblock.ss)
      return unnpfb(self._bblock.sequence)

   @property
   def pdbfile(self):
      return self.dbentry['file']

   @property
   def pdbkey(self):
      return self.bblock.filehash

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
      return bblock_is_cyclic(self._bblock)

   @property
   def repeataxis(self):
      return self._bblock.repeataxis

   @property
   def chains(self):
      return np.array(self._bblock.chains)

   @property
   def repeatstart(self):
      return self._bblock.repeatstart

   @property
   def repeatspacing(self):
      return self._bblock.repeatspacing

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

   @property
   def dbentry(self):
      return json.loads(self.json)

   # @property
   # def repeat_spacing(self):
   # return get_repeat_spacing(self)

   def make_extended_bblock(self, nrepeats=1, **kw):
      return make_extended_bblock(self, nrepeats, **kw)

   @property
   def stubs(self):
      return self._bblock.stubs

def bblock_is_cyclic(bblock):
   c = unnpfb(bblock.classes)
   if len(c) < 4: return False
   return all([
      c[0] == 'C',
      c[1].isdigit(),
      c[2] == '_',
      c[3] in 'CN',
   ])

def add_repeat_to_pose(pose, nrepeats, start, period, shift=10, **kw):
   from worms.rosetta_init import append_subpose_to_pose, append_pose_to_pose
   stubs, _ = worms.util.rosetta_utils.get_bb_stubs(pose)
   stub1 = stubs[start]
   stub2 = stubs[start + period]
   xrepeat = stub2 @ np.linalg.inv(stub1)
   for i in range(nrepeats):
      newpose = worms.rosetta_init.Pose()
      append_subpose_to_pose(newpose, pose, 1, start + period + shift, False)
      pose2 = pose.clone()
      worms.util.rosetta_utils.xform_pose(xrepeat, pose2)
      append_subpose_to_pose(newpose, pose2, start + shift + 1, pose2.size(), False)
      pose = newpose
   # tmp = worms.rosetta_init.Pose()
   # append_subpose_to_pose(tmp, pose2, start + shift + 1, pose2.size())
   # tmp.dump_pdb('new2.pdb')
   # pose.dump_pdb('orig.pdb')
   # newpose.dump_pdb('new.pdb')
   return newpose

def make_derived_bblock(pdbfile, bblock):
   pass

def make_extended_bblock(
   bblock,
   nrepeats=1,
   null_base_names=["", "?", "n/a", "none"],
   **kw,
):
   kw = wu.Bunch(kw)
   bblockdb = kw.database.bblockdb
   assert bblockdb is not None
   wu.PING(bblock.pdbfile)
   pose = bblockdb.pose(bblock.pdbfile)
   start, period = bblock.repeatstart, bblock.repeatspacing

   newpose = add_repeat_to_pose(pose, nrepeats, start, period, **kw)

   origentry = bblock.dbentry
   newentry = copy.copy(bblock.dbentry)
   newfile = add_props_to_url(bblock.pdbfile, addrepeat=nrepeats)
   newentry['file'] = newfile
   newentry['connections'] = list()
   for conn in origentry['connections']:
      conn2 = copy.copy(conn)
      if conn['direction'] == 'C':
         print(conn['residues'])
         conn2['residues'] = list()
         for c in conn['residues']:
            if isinstance(c, int):
               conn2['residues'].append(c + nrepeats * period)
            else:
               conn2['residues'].append(c)
      newentry['connections'].append(conn2)

   # support modifications in fname, probably in database?

   bblock2 = BBlock(make_bblock(newentry, newpose, null_base_names, **kw))
   # print(bblock.dbentry['connections'])
   # print(bblock2.dbentry['connections'])

   # assert 0
   return bblock2

def npfb(s):
   if isinstance(s, list):
      s = '[' + ','.join(s) + ']'
   return np.frombuffer(s.encode(), dtype='i1')

def unnpfb(fb):
   return bytes(fb).decode()

def get_repeat_axis(bblock, start, spacing, **kw):

   ncac1 = bblock.ncac[start:start + spacing, 1]
   ncac2 = bblock.ncac[start + spacing:start + 2 * spacing, 1]
   # import worms.viz.viz_bblock
   # wu.showme(bblock)
   # wu.showme(ncac1)
   # wu.showme(ncac2)
   coordsrepeataxisall = ncac2 - ncac1
   # print(coordsrepeataxisall)
   coordsrepeataxis = np.mean(coordsrepeataxisall, axis=0)
   # print(coordsrepeataxis)
   err = np.max(np.std(coordsrepeataxisall, axis=0))
   print(f'get_repeat_axis start {start} spacing {spacing} stddev {err}')
   # assert err < 1.0
   # print(coordsrepeataxis)

   return coordsrepeataxis, err

   #    repeataxisall = list()
   #    assert len(bblock.connections) == 2
   #    dirn1 = bblock.connections[0][0]
   #    dirn2 = bblock.connections[1][0]
   #    assert {dirn1, dirn2} == {0, 1}
   #    # for isite, (dirn, resi) in enumerate(bblock.connections):
   #    #    print(isite, dirn, resi)
   #    #    reslb, resub = np.min(resi), np.max(resi)
   #    #    nh, hrb, hre, hb, he = bblock.helixinfo(reslb, resub)
   #    #    if nh == 0: continue
   #    #    hcenters = (hb + he) / 2
   #    #    repeataxes = hcenters[2:] - hcenters[:-2]
   #    #    print('repeataxes\n', repeataxes)
   #    #    repeataxis = np.mean(repeataxes, axis=0)
   #    #    print('repeataxis', repeataxis)
   #    #    repeataxisall.append(repeataxis)
   #    # if len(repeataxisall) != 2:
   #    #    return np.array([0, 0, 0, 0], dtype=np.float32())
   #    # print(repeataxisall)
   #    # # repeataxis = hnormalized(repeataxisall[0] + repeataxisall[1]).astype(np.float32)
   #    # repeataxis = np.mean(repeataxes, axis=0)
   #    nh, hrb, hre, hb, he = bblock.helixinfo(100, len(bblock.ncac) - 100)
   #    for i, ir in enumerate(hrb):
   #       helixsep = ir - hrb[0]
   #       # print(i, helixsep)
   #       if abs(helixsep - spacing) < 5:
   #          break
   #    nhelixrepeat = i + 1
   #    print('nhelix per repeat', nhelixrepeat)
   #
   #    hcenters = (hb + he) / 2
   #    repeataxes = hcenters[2:] - hcenters[:-2]
   #    # repeataxes = hcenters[3:] - hcenters[:-3]
   #    print('repeataxes\n', repeataxes)
   #    repeataxis = np.mean(repeataxes, axis=0)
   #
   #    print('repeataxis', repeataxis)
   #
   #    return np.array([1, 0, 0, 0], dtype=np.float32)
   #    # return repeataxis

def get_repeat_spacing(bblock: BBlock, repeat_sequence_match_min_len=15, **kw):
   kw = wu.Bunch(kw)
   if bblock.is_cyclic: return None, None
   seq = bblock.sequence
   ss = bblock.ss
   N = len(seq)
   hnum, hlo, hhi, *_ = bblock.helixinfo()
   matcher = SequenceMatcher(None, seq, seq)
   spacings, starts = list(), list()
   for ih, (ilo, ihi) in enumerate(zip(hlo, hhi)):
      for jh, (jlo, jhi) in enumerate(zip(hlo, hhi)):
         if ih >= jh: continue
         a, b, n = matcher.find_longest_match(ilo, ihi, jlo, jhi)
         if n < repeat_sequence_match_min_len: continue
         # print('match', a, b, seq[a:a + n], seq[b:b + n])
         spacing = b - a
         starts.append(a)
         for s in spacings:
            spacing = min(spacing, math.gcd(s, spacing))
         spacings.append(spacing)
   # start = int(statistics.median(starts))
   if len(spacings) < 4 or len(set(spacings)) != 1:
      print('bblock.py get_repeat_spacing fail, spacings:', spacings)
      # assert 0
      return None, None

   start = min(starts)
   if len(starts) > 2:
      start = int(statistics.median(starts))
   spacing = spacings[0]

   stub0 = bblock.stubs[start]
   stub1 = bblock.stubs[start + spacing]
   stub2 = bblock.stubs[start + 2 * spacing]
   x1ang = wu.hangle_of_degrees(stub1 @ np.linalg.inv(stub0))
   x2ang = wu.hangle_of_degrees(stub2 @ np.linalg.inv(stub0))
   if x1ang > kw['repeat_twist_tolerance']:
      print('bblock.py:get_repeat_spacing xform for one repeat not identity', x1ang)
      # check if two repeats to make straight
      if x2ang < kw['repeat_twist_tolerance']:
         spacing *= 2
      else:
         print('bblock.py:get_repeat_spacing xform for two repeats not identity', x2ang)
         return None, None

   return start, spacing
