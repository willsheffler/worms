import tempfile, sys
import numpy as np
import willutil as wu
from willutil.homog.hgeom import hxform
import worms
from worms.homog.hgeom import hnormalized

if 'pymol' in sys.modules:

   import pymol

   @wu.viz.pymol_viz.pymol_load.register(worms.bblock.BBlock)
   def pymol_load_BBlock(
      bblock,
      name='unk',
      pos=np.eye(4),
      delprev=False,
      # resrange=(0, -1),
      # sym=None,
      # showframes=False,
      # nbrs=None,
      suspend_updates=True,
      scale=1.0,
      segcol=[1, 1, 1],
      # markfirst=False,
      addtocgo=None,
      insite=None,
      inchain=None,
      inres=None,
      outsite=None,
      outchain=None,
      outres=None,
      showextras=False,
      showsymaxis=False,
      **kw,
   ):
      from pymol import cmd
      if suspend_updates: cmd.set('suspend_updates', 'on')
      if delprev: cmd.delete(f'{name}*')
      name = 'wbb_' + name

      # pos0 = pos
      pos = pos.reshape(4, 4)
      # bbsnfold = len(body) // len(body.asym_body)
      # breaks = bbsnfold * len(pos) * len(sym)

      # breaks_groups = bbsnfold
      # print(breaks, breaks_groups)

      # print(pos.shape)
      coord = wu.hxform(pos, np.array(bblock.ncac))
      kw['name'] = name

      mycgo = list()

      for ichain, (reslb, resub) in enumerate(bblock.chains):
         col = np.eye(3)[ichain]
         # print('chain', ichain, reslb, resub, col)
         wu.viz.show_ndarray_line_strip(coord[reslb:resub], col=col, addtocgo=mycgo, **kw)

      if showsymaxis:
         if bblock.is_cyclic:
            com = wu.hxform(pos, bblock.com)
            # print(pos[:, 3])
            # print(com)
            # assert 0

            axlen = 200
            symaxis = wu.hxform(pos, [0, 0, 1, 0])
            symaxcen = com - axlen / 2 * symaxis
            symaxvec = axlen * symaxis
            symaxray = np.stack([symaxcrepeataxisen, symaxvec], axis=1)
            wu.viz.show_ndarray_lines(symaxray, col=[0, 0, 0.5], cyl=1, spheres=1.3,
                                      bothsides=False, addtocgo=mycgo, **kw)

      # nh, hrb, hre, hb, he = worms.vertex.get_bb_helices(bblock._bblock, **kw)
      # print(bblock.ss)
      if showextras and not bblock.is_cyclic:
         repeataxisall = list()
         for isite, (dirn, resi) in enumerate(bblock.connections):

            reslb, resub = np.min(resi), np.max(resi)
            nh, hrb, hre, hb, he = bblock.helixinfo(reslb, resub)
            hb = wu.hxform(pos, hb)
            he = wu.hxform(pos, he)

            hcenters = (hb + he) / 2
            repeataxes = hcenters[2:] - hcenters[:-2]
            hcenbeg = hcenters[:-2] + repeataxes * 0.3
            hcendir = repeataxes * 0.4
            repeatrays = np.moveaxis(np.array([hcenbeg, hcendir]), [0, 1, 2], [2, 0, 1])
            repeataxis = np.mean(repeataxes, axis=0)
            repeataxisall.append(repeataxis)
            rdir = repeataxis * len(hcenters) / 4
            rcen = np.mean(hcenters, axis=0) - rdir / 2
            repeatrays2 = np.stack([rcen, rdir], axis=1)

            wu.viz.show_ndarray_lines(repeatrays, col=[0, 0, 0.5], cyl=0.5, spheres=0,
                                      bothsides=False, addtocgo=mycgo, **kw)

            wu.viz.show_ndarray_lines(repeatrays2, col=[0, 0.5, 0.5], cyl=0.8, spheres=1.0,
                                      bothsides=False, addtocgo=mycgo, **kw)

            hrays = np.moveaxis(np.array([hb, he - hb]), [0, 1, 2], [2, 0, 1])
            wu.viz.show_ndarray_lines(hrays, col=[0.8, 0, 0.8], cyl=0.9, spheres=1.1,
                                      bothsides=False, addtocgo=mycgo, **kw)

            # wu.viz.show_ndarray_point_or_vec(hcenters, col=[0.5, 0, 0.5], sphere=0.5,
            # addtocgo=mycgo, **kw)3

            repeatrays3 = np.stack([rcen, bblock.repeataxis], axis=1)
            wu.viz.show_ndarray_lines(repeatrays3, col=[1, 1, 1], cyl=1.2, spheres=1.5,
                                      bothsides=False, addtocgo=mycgo, **kw)

         # assert len(repeataxisall) == 2
         # repeataxis = hnormalized(repeataxisall[0] + repeataxisall[1])
         # rdir = repeataxis * 100
         # tmpray = np.stack([hxform(pos, bblock.com) - rdir / 2, rdir], axis=1)
         # wu.viz.show_ndarray_lines(tmpray, col=[1, 1, 1], cyl=1.2, spheres=1.8, bothsides=False,
         # addtocgo=mycgo, **kw)

      if addtocgo is not None: addtocgo.extend(mycgo)
      else: pymol.cmd.load_cgo(mycgo, name)
      if suspend_updates: cmd.set('suspend_updates', 'off')
