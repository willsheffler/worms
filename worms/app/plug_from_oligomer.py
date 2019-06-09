import sys, os, logging, blosc, pyrosetta, concurrent.futures
import numpy as np
import sicdock
from sicdock import Timer, Bunch, util as sutil
from sicdock.motif import HierScore
from sicdock.search import grid_search, make_plugs
from sicdock.sampling import grid_sym_axis
# from sicdock.util import NOCACHE as cache
from sicdock.util import GLOBALCACHE as cache
import worms

from worms import prune_clashes

def main():
   arg, criteria = setup()
   arg.plug_fixed_olig = True
   # arg.executor = concurrent.futures.ThreadPoolExecutor(2)

   if arg.run_cache and os.path.exists(arg.run_cache):
      cache.load(arg.run_cache, strict=False)

   ssdag, wresult = worms_search(criteria, **arg)
   if len(wresult.idx) == 0:
      logging.info('no results')
      return
   logging.info(f'worms results {wresult.idx.shape}')

   results = plug_dock(wresult, ssdag, criteria, **arg)
   logging.info(f'n results {results.ndocks}')

   if arg.dump_pdbs:
      results.dump_pdbs_top_score(**arg)
      results.dump_pdbs_top_score_each(**arg)
   if not arg.suppress_dump_results:
      sutil.dump(results, arg.output_prefix + '_Result.pickle')

   shutdown(**arg)

def setup():
   prof = Timer(verbose=False).start()
   pyrosetta.init("-mute all -beta -preserve_crystinfo --prevent_repacking")
   blosc.set_releasegil(True)
   prof.checkpoint('pyrosetta init')

   # parser = sicdock.app.default_cli_parser()
   # critlist, kw = worms.cli.build_worms_setup_from_cli_args(sys.argv[1:], parser)
   worms_parser = worms.cli.make_cli_arg_parser()
   parser = sicdock.app.default_cli_parser(parent=worms_parser)
   critlist, kw = worms.cli.build_worms_setup_from_cli_args(sys.argv[1:], parser)

   criteria = critlist[0]
   arg = sicdock.app.process_cli_args(kw)
   del kw
   arg.prof = prof

   bbtype = criteria.bbspec[0][0]
   arg.sym = bbtype[:2]
   if not (bbtype[0] == 'C' and bbtype[2] == '_'):
      raise ValueError("can'prof figure out what sym should be")
   arg.nfold = int(arg.sym[1])
   arg.prof.checkpoint('setup')

   return arg, criteria

def worms_search(criteria, **kw):
   arg = Bunch(kw)
   ssdag = cache(worms.simple_search_dag, criteria, _key='ssdag', **arg)
   arg.prof.checkpoint('worms dag')
   wresult = cache(worms.grow_linear, ssdag, criteria.jit_lossfunc(), _key='wresult', **arg)
   arg.prof.checkpoint('worms search')
   # wresult = cache(worms.prune_clashes, ssdag, criteria, wresult, **arg)
   arg.prof.checkpoint('worms clash')
   return ssdag, wresult

def shutdown(prof, run_cache, **kw):
   prof.stop().report()
   cache.save(run_cache)

def plug_dock(wresult, ssdag, criteria, max_dock=-1, **kw):
   arg = Bunch(kw)

   hscore = cache(HierScore, arg.hscore_files, _nodump=True)
   arg.prof.checkpoint('load hscore')

   hole = cache(sicdock.body.Body, arg.context_structure, sym=arg.sym, which_ss='H')
   # hole.dump_pdb(os.path.basename(arg.context_structure) + "_hole.pdb")
   hole.dump_pdb("context_structure.pdb")
   arg.prof.checkpoint('make hole')

   crt_smap = np.arange(-100, 100.1, arg.grid_resolution_cart_angstroms)
   ori_samp = np.arange(-180 / arg.nfold, 180 / arg.nfold - 0.001,
                        arg.grid_resolution_ori_degrees)
   samples = grid_sym_axis(crt_smap, ori_samp, axis=[0, 0, 1], flip=[0, 1, 0])
   logging.info(f'docking samples per splice {len(samples)}')

   enddir = dict(N='C', C='N')[criteria.bbspec[-1].direction[0]]
   results = list()

   futures = list()
   exe = worms.util.InProcessExecutor()
   if arg.parallel > 1:
      exe = concurrent.futures.ProcessPoolExecutor(arg.parallel)

   with exe as pool:
      for iresult, (idx, pos) in enumerate(zip(wresult.idx, wresult.pos)):
         if max_dock > 0 and iresult >= max_dock:
            break
         pose = worms.make_pose_crit(arg.db[0], ssdag, criteria, idx, pos)
         arg.prof.checkpoint('make pose')
         label, bbnames = make_label(ssdag, idx, **arg)
         futures.append(
            pool.submit(plug_dock_one, hole, samples, pose, label, bbnames, enddir, iresult,
                        **arg.sub(db=None, dont_store_body_in_results=True)))
      results, plugs, profs = zip(*[f.result() for f in futures])
   arg.prof.merge(profs)

   results = sicdock.search.concat_results(results)
   if not arg.dont_store_body_in_results:
      results.body_labels = 'plug hole'.split()
      results.bodies = [(p, hole) for p in plugs]

   return results

def plug_dock_one(hole, samples, pose, label, bbnames, enddir, iresult, **kw):
   arg = Bunch(kw)
   prof = Timer().start()
   plug = sicdock.body.Body(pose, which_ss="H", trim_direction=enddir, label=label,
                            components=bbnames)
   prof.checkpoint('make body')
   op = f'{arg.output_prefix}_{iresult:04}'
   hscore = cache(HierScore, arg.hscore_files)  # read from forked data
   result = make_plugs(plug, hole, hscore, samples, grid_search, **arg.sub(output_prefix=op))
   prof.checkpoint('plug dock')
   return result, plug, prof

def make_label(ssdag, idx, sep='__', **kw):
   bbs = [ssdag.bbs[i][ssdag.verts[i].ibblock[idx[i]]] for i in range(len(idx))]
   names = [bytes(b.name).decode() for b in bbs]
   splice = [ssdag.verts[i].ires[idx[i]] for i in range(len(idx))]
   splice = np.concatenate(splice)
   splice = np.array([s for s in splice if s >= 0])
   splice = splice.reshape(-1, 2)
   ndigit = sutil.num_digits(max(np.max(v.ires) for v in ssdag.verts))
   s = str()
   for i, n in enumerate(names):
      if i > 0:
         s += f'_r{splice[i-1,0]:0{ndigit}}{sep}r{splice[i-1,1]:0{ndigit}}_'
      s += n
   return f'_{s}_', names

def main_test_dump():
   arg, crit = setup()
   result = sicdock.load('worms_Result.pickle')
   assert isinstance(result, sicdock.Result)
   arg.dump_pbds = True
   result.dump_pdbs_top_score(**arg)
   result.dump_pdbs_top_score_each(**arg)

if __name__ == '__main__':
   main_test_dump()
   # main()
