import string
import numpy as np
import worms

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
      chain=0,
      anum=1,
      rnum=1,
):

   close = False
   if isinstance(out, str):
      out = open(out, "w")
      close = True

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
      # chains will have insplice at first ops, outsplice at last pos
      # either could be none
      chains = list()
      chains.append(sponly[0] if dirn[0] < 2 else None)
      for c in chains0:
         if np.all(sponly[0] == c) or np.all(sponly[-1] == c):
            continue
         chains.append(c)
      if len(sponly) > 1 or chains[0] is None:
         chains.append(sponly[-1] if dirn[1] < 2 else None)

   aname = [" N  ", " CA ", " C  "]
   for ic, lbub in enumerate(chains):
      if lbub is None:
         continue
      for i in range(*lbub):
         for j in (0, 1, 2):
            xyz = pos @ bblock.ncac[i, j]
            out.write(
               format_atom(
                  atomi=anum,
                  atomn=aname[j],
                  resn="GLY",
                  chain=string.ascii_uppercase[chain],
                  resi=rnum,
                  x=xyz[0],
                  y=xyz[1],
                  z=xyz[2],
                  occ=1.0,
               ))
            anum += 1
         rnum += 1
      if join == "bb":
         continue
      if join == "splice" and ic + 1 == len(chains) and dirn[1] < 2:
         continue
      chain += 1
   if join == "bb":
      chain += 1
   if close:
      out.close()
   return chain, anum, rnum

def graph_dump_pdb(
      out,
      ssdag,
      idx,
      pos,
      join="splice",
      xalign=np.eye(4),
      trim=True,
      crystinfo=None,
):
   # print('graph_dump_pdb')

   close = False
   if isinstance(out, str):
      out = open(out, "w")
      close = True

   # print(crystinfo, flush=True)

   cryst1 = 'CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 ' % crystinfo[:3] + crystinfo[6]
   out.write(cryst1 + '\n')

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
         pos=xalign @ x,
         chain=chain,
         anum=anum,
         rnum=rnum,
         join=join,
      )
   if close:
      out.close()
