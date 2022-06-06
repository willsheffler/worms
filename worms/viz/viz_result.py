import tempfile, sys, copy
import numpy as np
import willutil as wu
import worms

if 'pymol' in sys.modules:

   import pymol

   @wu.viz.pymol_viz.pymol_load.register(worms.search.result.ResultTable)
   def pymol_load_Result(
      result,
      iresult,
      name='unk',
      delprev=False,
      # resrange=(0, -1),
      sym=None,
      palette=None,
      # showframes=False,
      # nbrs=None,
      suspend_updates=True,
      addtocgo=None,
      **kw,
   ):
      from pymol import cmd
      if suspend_updates: cmd.set('suspend_updates', 'on')
      if delprev: cmd.delete(f'{name}*')
      name = 'wresult_' + name
      palette = wu.viz.pymol_viz.get_palette(palette)

      ssdag = result.ssdag
      verts = result.ssdag.verts

      # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!! second !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      # worms.viz.viz_bblock.pymol_load_BBlock(ssdag.bblocks[2][1], name='second')

      segpos = result.pos[iresult]
      xalign = result.criteria.alignment(segpos)

      mycgo = list()

      for iseg, v in enumerate(verts):
         ivert = result.idx[iresult, iseg]
         pos = xalign @ result.pos[iresult, iseg]
         assert pos.shape == (4, 4)
         ibb = v.ibblock[ivert]
         bb = ssdag.bbs[iseg][ibb]
         fname = str(bytes(bb.file), 'utf-8')
         inres = v.ires[ivert, 0]
         outres = v.ires[ivert, 1]
         insite = v.isite[ivert, 0]
         outsite = v.isite[ivert, 1]
         inchain = v.ichain[ivert, 0]
         outchain = v.ichain[ivert, 1]
         dirn = v.dirn
         if fname.startswith(r'pdbs\\\\'):
            fname = fname[4:].replace('\\\\', '/')
         print(
            f'{iseg:3} ',
            f'{ivert:6} ',
            f'{ibb:2}',
            f'{v.dirn} ',
            f'{insite:3} ',
            f'{inchain:3} ',
            f'{inres:4} ',
            '/ '
            f'{outsite:3} '
            f'{outchain:3} ',
            f'{outres:4} ',
         )

         # dbentry = copy.deepcopy(database.bblockdb._dictdb[fname])
         symframes = np.eye(4).reshape(1, 4, 4)
         if isinstance(sym, str):
            symframes = wu.sym.frames(sym)
         elif isinstance(sym, np.ndarray):
            symframes = sym
         else:
            assert sym is None

         for isym, xsym in enumerate(symframes):
            worms.viz.viz_bblock.pymol_load_BBlock(
               ssdag.bblocks[iseg][ibb],
               name=f'worms.BBlock_{iseg}_{isym}',
               # pos=result.pos[iresult][iseg],
               pos=xsym @ pos,
               delprev=False,
               # resrange=(0, -1),
               # sym=sym,
               # showframes=False,
               # nbrs=None,
               suspend_updates=False,
               scale=1.0,
               bounds=(0, -1),
               segcol=palette[iseg],
               # markfirst=False,
               addtocgo=mycgo,
               insite=insite,
               inchain=inchain,
               inres=inres,
               outsite=outsite,
               outchain=outchain,
               outres=outres,
               showsymaxis=True,
               **kw,
            )

      if addtocgo is not None: addtocgo.extend(mycgo)
      else: pymol.cmd.load_cgo(mycgo, name)
      if suspend_updates: cmd.set('suspend_updates', 'off')
