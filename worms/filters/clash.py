import concurrent.futures as cf
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm

from worms.util import priority_jit, InProcessExecutor
from worms.search.result import ResultJIT
from worms.clashgrid import ClashGrid

def prune_clashes(
   ssdag,
   crit,
   rslt,
   max_clash_check=-1,
   ca_clash_dis=4.0,
   parallel=False,
   approx=0,
   verbosity=0,
   merge_bblock=None,
   pbar=False,
   pbar_interval=10.0,
   context_structure=None,
   **kw,
):
   # print('todo: clash check should handle symmetry')
   if max_clash_check == 0:
      return rslt
   max_clash_check = min(max_clash_check, len(rslt.idx))
   if max_clash_check < 0:
      max_clash_check = len(rslt.idx)

   if not pbar:
      print(
         f"mbb{f'{merge_bblock:04}' if merge_bblock else 'none'} checking clashes",
         max_clash_check,
         "of",
         len(rslt.err),
      )

   verts = tuple(ssdag.verts)
   # exe = cf.ProcessPoolExecutor if parallel else InProcessExecutor
   exe = InProcessExecutor
   with exe() as pool:
      futures = list()
      for i in range(max_clash_check):
         dirns = tuple([v.dirn for v in verts])
         iress = tuple([v.ires for v in verts])
         chains = tuple([
            ssdag.bbs[k][verts[k].ibblock[rslt.idx[i, k]]].chains for k in range(len(ssdag.verts))
         ])
         ncacs = tuple([
            ssdag.bbs[k][verts[k].ibblock[rslt.idx[i, k]]].ncac for k in range(len(ssdag.verts))
         ])
         if isinstance(context_structure, ClashGrid):
            clash = False
            for pos, ncac in zip(rslt.pos[i], ncacs):
               xyz = pos @ ncac[..., None]
               if context_structure.clashcheck(xyz.squeeze()):
                  clash = True
                  break
            if clash:
               continue

         futures.append(
            pool.submit(
               _check_all_chain_clashes,
               dirns=dirns,
               iress=iress,
               idx=rslt.idx[i],
               pos=rslt.pos[i],
               chn=chains,
               ncacs=ncacs,
               thresh=ca_clash_dis * ca_clash_dis,
               approx=approx,
            ))
         futures[-1].index = i

      if pbar:
         desc = "checking clashes "
         if merge_bblock is not None and merge_bblock >= 0:
            desc = f"{desc}    mbb{merge_bblock:04d}"
         if merge_bblock is None:
            merge_bblock = 0
         futures = tqdm(
            cf.as_completed(futures),
            desc=desc,
            total=len(futures),
            mininterval=pbar_interval,
            position=merge_bblock + 1,
         )

      ok = np.zeros(max_clash_check, dtype="?")
      for f in futures:
         ok[f.index] = f.result()

   return ResultJIT(
      rslt.pos[:max_clash_check][ok],
      rslt.idx[:max_clash_check][ok],
      rslt.err[:max_clash_check][ok],
      rslt.stats,
   )

@priority_jit
def _chain_bounds(dirn, ires, chains, spliced_only=False, trim=8):
   "return bounds for only spliced chains, with spliced away sequence removed"
   chains = np.copy(chains)
   bounds = []
   seenchain = -1
   if dirn[0] < 2:
      ir = ires[0]
      for i in range(len(chains)):
         lb, ub = chains[i]
         if lb <= ir < ub:
            chains[i, dirn[0]] = ir + trim * (1, -1)[dirn[0]]
            bounds.append((chains[i, 0], chains[i, 1]))
            seenchain = i
   if dirn[1] < 2:
      ir = ires[1]
      for i in range(len(chains)):
         lb, ub = chains[i]
         if lb <= ir < ub:
            chains[i, dirn[1]] = ir + trim * (1, -1)[dirn[1]]
            if seenchain == i:
               if dirn[1]:
                  tmp = bounds[0][0], chains[i, 1]
               else:
                  tmp = chains[i, 0], bounds[0][1]
               # bounds[0][dirn[1]] = chains[i, dirn[1]]
               bounds[0] = tmp
            else:
               bounds.append((chains[i, 0], chains[i, 1]))
   if spliced_only:
      return np.array(bounds, dtype=np.int32)
   else:
      return chains

@priority_jit
def _has_ca_clash(position, ncacs, i, ichntrm, j, jchntrm, thresh, step=1):
   for ichain in range(len(ichntrm)):
      ilb, iub = ichntrm[ichain]
      for jchain in range(len(jchntrm)):
         jlb, jub = jchntrm[jchain]
         for ir in range(ilb, iub, step):
            ica = position[i] @ ncacs[i][ir, 1]
            for jr in range(jlb, jub, step):
               jca = position[j] @ ncacs[j][jr, 1]
               d2 = np.sum((ica - jca)**2)
               if d2 < thresh:
                  return True
   return False

@priority_jit
def _check_all_chain_clashes(dirns, iress, idx, pos, chn, ncacs, thresh, approx):
   pos = pos.astype(np.float32)

   for step in (3, 1):  # 20% speedup.... ug... need BVH...

      # only adjacent verts, only spliced chains
      for i in range(len(dirns) - 1):
         ichn = _chain_bounds(dirns[i], iress[i][idx[i]], chn[i], 1, 8)
         for j in range(i + 1, i + 2):
            jchn = _chain_bounds(dirns[j], iress[j][idx[j]], chn[j], 1, 8)
            if _has_ca_clash(pos, ncacs, i, ichn, j, jchn, thresh, step):
               return False
      if step == 1 and approx == 2:
         return True

      # only adjacent verts, all chains
      for i in range(len(dirns) - 1):
         ichn = _chain_bounds(dirns[i], iress[i][idx[i]], chn[i], 0, 8)
         for j in range(i + 1, i + 2):
            jchn = _chain_bounds(dirns[j], iress[j][idx[j]], chn[j], 0, 8)
            if _has_ca_clash(pos, ncacs, i, ichn, j, jchn, thresh, step):
               return False
      if step == 1 and approx == 1:
         return True

      # all verts, all chains
      for i in range(len(dirns) - 1):
         ichn = _chain_bounds(dirns[i], iress[i][idx[i]], chn[i], 0, 8)
         for j in range(i + 1, len(dirns)):
            jchn = _chain_bounds(dirns[j], iress[j][idx[j]], chn[j], 0, 8)
            if _has_ca_clash(pos, ncacs, i, ichn, j, jchn, thresh, step):
               return False

   return True
