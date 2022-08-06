import string
import numpy as np
from willutil.sym import SymFit
import worms
import willutil as wu

_atom_record_format = (
   "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
   "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}\n")

def format_atom(
   atomi=0,
   atomn="ATOM",
   idx=" ",
   resn="RES",
   chain="A",
   resi=0,
   insert=" ",
   x=0,
   y=0,
   z=0,
   occ=0,
   b=0,
):
   return _atom_record_format.format(**locals())

def bblock_dump_pdb(
   out,
   bblock,
   dirn=(2, 2),
   splice=(-1, -1),
   join="splice",
   pos=np.eye(4),
   cnum=0,
   anum=1,
   rnum=1,
   extrachains=False,
   chainlabels=None,
   extension=0,
   sym='c1',
   bblockdb=None,
   **kw,
):
   'dirn=(2, 2), # 2 == terminus      splice=(-1, -1), # -1 is no splice'
   close = False
   if isinstance(out, str):
      out = open(out, "w")
      close = True
   symframes = wu.sym.frames(sym)
   if len(symframes) > 1 and extrachains:
      print(f'bblock_dump_pdb symmetric {sym} output disables "extrachains"')
      extrachains = False

   if not chainlabels:
      chainlabels = string.ascii_uppercase + string.ascii_uppercase + '0123456789'

   chains0 = worms.filters.clash._chain_bounds(
      dirn,
      splice,
      bblock.chains,
      trim=0,
   )
   if dirn[0] == 2 and dirn[1] == 2:
      chains = chains0
   else:
      sponly = worms.filters.clash._chain_bounds(
         dirn,
         splice,
         bblock.chains,
         trim=0,
         spliced_only=1,
      )
      # chains will have insplice at first pos, outsplice at last pos
      # either could be none
      chains = list()
      chains.append(sponly[0] if dirn[0] < 2 else None)
      for c in chains0:
         if np.all(sponly[0] == c) or np.all(sponly[-1] == c):
            continue
         chains.append(c)
      if len(sponly) > 1 or chains[0] is None:
         chains.append(sponly[-1] if dirn[1] < 2 else None)

   nres_extension = 0
   if extension:
      print('??????????????????????????????', extension)
      print('extension ncac', bblock.ncac.shape)
      bblock = worms.bblock.make_extended_bblock(bblock, nrepeats=extension, bblockdb=bblockdb,
                                                 **kw)
      print('extension ncac', bblock.ncac.shape)
      start = bblock.repeatstart
      nres_extension = bblock.repeatspacing
      nres_extension *= extension
      print('extension', start, nres_extension)
      # assert 0

      print('??????????????????????????????')

   mnconn, mxconn = 9e9, 0
   for ich, conn in bblock.connections:
      mnconn = min(mnconn, np.min(conn))
      mxconn = max(mnconn, np.max(conn))

   for isym, xsym in enumerate(symframes):
      aname = [' N  ', ' CA ', ' C  ']
      for ichain, lbub in enumerate(chains):
         if lbub is None:
            continue
         # print('lbub', lbub, nres_extension)
         lbub = lbub[0], lbub[1] + nres_extension
         # print('lbub2', lbub)

         if not extrachains and bblock.is_cyclic:
            # print('extrachains', ichain, lbub)
            if lbub[0] > mxconn or lbub[1] < mnconn:
               # print('dumpdb.py skip', ichain, lbub)
               continue

         # ATOM      2  CA  GLY A   1    -293.330  -6.986-106.420  1.00  0.00
         # ATOM  20412  C   GLY 38040      99.531 168.935-274.593  1.00  0.00

         for ires in range(*lbub):
            for iatom in (0, 1, 2):
               xyz = xsym @ pos @ bblock.ncac[ires, iatom]
               out.write(
                  format_atom(
                     atomi=anum % 100_000,
                     atomn=aname[iatom],
                     resn="GLY",
                     chain=chainlabels[cnum % len(chainlabels)],
                     resi=rnum % 10_000,
                     x=xyz[0],
                     y=xyz[1],
                     z=xyz[2],
                     occ=1.0,
                  ))
               anum += 1
            rnum += 1
         if join == "bb":
            continue
         if join == "splice" and ichain + 1 == len(chains) and dirn[1] < 2:
            continue
         cnum += 1
      if join == "bb":
         cnum += 1

   if close:
      out.close()
   return cnum, anum, rnum

def graph_dump_pdb(
      out,
      ssdag,
      idx,
      pos,
      join="splice",
      xalign=np.eye(4),
      trim=True,
      crystinfo=None,
      extensions=dict(),
      sym='c1',
      **kw,
):
   # print('graph_dump_pdb')
   kw = wu.Bunch(kw)

   bblockdb = kw.database.bblockdb

   close = False
   if isinstance(out, str):
      out = open(out, "w")
      close = True

   # print(crystinfo, flush=True)

   bblocks = [
      ssdag.bblocks[iseg][ssdag.verts[iseg].ibblock[idx[iseg]]] for iseg in range(len(idx))
   ]

   xalign, pos = worms.extension.modify_xalign_cage_by_extension(
      ssdag=ssdag,
      idx=idx,
      pos=pos,
      xalign=xalign,
      bblockdb=bblockdb,
      extensions=extensions,
      bblocks=bblocks,
      **kw,
   )

   if crystinfo:
      cryst1 = 'CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 ' % crystinfo[:3] + crystinfo[6]
      out.write(cryst1 + '\n')

   assert len(idx) == len(pos)
   assert idx.ndim == 1
   assert pos.ndim == 3
   assert pos.shape[-2:] == (4, 4)
   cnum, anum, rnum = 0, 1, 1

   for i, (bbs, vert, ivert, x) in enumerate(zip(ssdag.bblocks, ssdag.verts, idx, pos)):

      # derived bblocks here
      extension = extensions.get(i, 0)

      cnum, anum, rnum = bblock_dump_pdb(
         out=out,
         bblock=bbs[vert.ibblock[ivert]],
         dirn=vert.dirn if trim else (2, 2),
         splice=vert.ires[ivert] if trim else (-1, -1),
         pos=xalign @ x,
         cnum=cnum,
         anum=anum,
         rnum=rnum,
         join=join,
         extension=extension,
         bblockdb=bblockdb,
         sym=sym,
         **kw,
      )
   if close:
      out.close()
