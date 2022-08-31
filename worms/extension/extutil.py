import worms
import willutil as wu
import numpy as np

def modify_xalign_cyclic_by_extension(
   ssdag,
   idx,
   pos,
   xalign,
   bblocks,
   bblockdb,
   extensions,
   **kw,
):
   if not extensions or set(extensions.values()) == {0}:
      return xalign, pos

   extpos = get_extended_pos(ssdag, idx, pos, xalign, bblocks, bblockdb, extensions, **kw)
   x_from = xalign @ extpos[0]
   x_to = xalign @ extpos[-1]
   xhat = x_to @ wu.hinv(x_from)
   axis, ang, cen = wu.haxis_ang_cen_of(xhat)

   # print(axis, ang, cen)
   # assert 0

   # trans = wu.htrans(cen - xalign[:, 3])
   trans = wu.htrans(-cen)
   xalign = trans @ xalign
   # print(cen)

   # print(xalign)
   return xalign, extpos

def modify_xalign_cage_by_extension(
   ssdag,
   idx,
   pos,
   xalign,
   bblocks,
   bblockdb,
   extensions,
   **kw,
):
   if not extensions or set(extensions.values()) == {0}:
      return xalign, pos

   extpos = get_extended_pos(ssdag, idx, pos, xalign, bblocks, bblockdb, extensions, **kw)

   cyc0 = bblocks[0].classes.split('_')[0]
   cyc1 = bblocks[-1].classes.split('_')[0]
   ax0 = alnpos[0, :, 2]
   ax1 = alnpos[-1, :, 2]
   # print(cyc0, ax0)
   # print(cyc1, ax1)
   repeataxis = segpos @ bblock.repeataxis

   # print(repeataxis)
   shift = wu.homog.proj_perp(ax1, repeataxis)
   cros = wu.homog.hcross(ax0, ax1)
   # print(cros)
   shift = wu.homog.proj_perp(cros, shift)
   assert np.allclose(wu.hdot(cros, shift), 0)
   # print('shift', shift)

   # symax = wu.sym.axes(sym)
   # print(symax)
   # assert 0

   shiftmag = wu.hnorm(shift) / np.sin(wu.hangle(ax0, ax1))
   if wu.hdot(xalign[:, 3], ax0) < 0:
      shiftmag = -shiftmag
   xextaln = wu.htrans(ax0 * shiftmag * nrepeat)

   xalign = xextaln @ xalign

   return xalign, extpos

   # assert 0

def get_extended_pos(
   ssdag,
   idx,
   pos,
   xalign,
   bblocks,
   bblockdb,
   extensions,
   **kw,
):
   pos = pos.copy()
   alnpos = xalign @ pos

   assert len(extensions) == 1
   iseg = list(extensions.keys())[0]
   segpos = alnpos[iseg]
   nrepeat = list(extensions.values())[0]

   bblock = ssdag.bblocks[iseg][ssdag.verts[iseg].ibblock[idx[iseg]]]
   bblock = worms.bblock.make_extended_bblock(bblock, nrepeats=nrepeat, bblockdb=bblockdb, **kw)
   start = bblock.repeatstart
   nres = bblock.repeatspacing
   # print(start, nres)
   # point1 = segpos @ bblock.ncac[start, 1]
   # point2 = segpos @ bblock.ncac[start + nrepeat * nres, 1]
   # print('direction', point2 - point1)
   # print('arst', bblock.stubs.shape)
   stub0 = segpos @ bblock.stubs[start]
   stub1 = segpos @ bblock.stubs[start + nrepeat * nres]
   xextend = stub1 @ np.linalg.inv(stub0)
   # print(stub0.shape)
   # print(stub1.shape)
   # print('xextend')
   # print(xextend)
   # print(segpos)
   for iseg2 in range(iseg + 1, len(idx)):
      pos[iseg2] = np.linalg.inv(xalign) @ xextend @ xalign @ pos[iseg2]

   return pos