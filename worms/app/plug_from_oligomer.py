import sys, os, logging, blosc, pyrosetta, concurrent.futures
import numpy as np, rpxdock as rp
from rpxdock.search import make_plugs
# from rpxdock.util import NOCACHE as cache
from rpxdock.util import GLOBALCACHE as cache
import worms
from worms import rosetta_init
from worms.filters.db_filters import get_affected_positions
from worms.rosetta_init import rosetta_init_safe
from willutil import bunch

def main():
   arg, criteria = setup()
   arg.plug_fixed_olig = True
   arg.output_body = 0
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
      rp.util.dump(results, arg.output_prefix + '_Result.pickle')

   shutdown(**arg)

def setup():
   prof = rp.Timer(verbose=False).start()
   rosetta_init_safe("-mute all -beta -preserve_crystinfo --prevent_repacking")
   blosc.set_releasegil(True)
   prof.checkpoint('pyrosetta init')

   # parser = rp.app.default_cli_parser()
   # critlist, kw = worms.cli.build_worms_setup_from_cli_args(sys.argv[1:], parser)
   worms_parser = worms.cli.make_cli_arg_parser()
   parser = rp.app.default_cli_parser(parent=worms_parser)
   parser.add_argument("--dont_store_plugs", action='store_true', default=False)
   critlist, kw = worms.cli.build_worms_setup_from_cli_args(sys.argv[1:], parser)

   criteria = critlist[0]
   arg = rp.app.process_cli_args(kw)
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
   ssdag, _ = cache(worms.simple_search_dag, criteria, _key='ssdag', **arg)
   arg.prof.checkpoint('worms dag')
   wresult = cache(worms.grow_linear, ssdag, criteria.jit_lossfunc(), _key='wresult', **arg)
   arg.prof.checkpoint('worms search')
   # wresult = cache(worms.filters.clash.prune_clashes, ssdag, criteria, wresult, **arg)
   arg.prof.checkpoint('worms clash')
   return ssdag, wresult

def shutdown(prof, run_cache, **kw):
   prof.stop().report()
   cache.save(run_cache)

def plug_dock(wresult, ssdag, criteria, max_dock=-1, **kw):
   arg = Bunch(kw)

   hscore = cache(rp.RpxHier, arg.hscore_files, hscore_data_dir=arg.hscore_data_dir, _nodump=True)
   arg.prof.checkpoint('load hscore')

   hole = cache(rp.body.Body, arg.context_structure, sym=arg.sym, which_ss='H')
   # hole.dump_pdb(os.path.basename(arg.context_structure) + "_hole.pdb", use_body_sym=True)
   hole.dump_pdb("context_structure.pdb", use_body_sym=True)
   arg.prof.checkpoint('make hole')

   cb = arg.cart_bounds[0]
   if not cb: cb = [-100, 100]
   if arg.docking_method.lower() == 'grid':
      search = rp.grid_search
      crt_smap = np.arange(cb[0], cb[1] + 0.001, arg.grid_resolution_cart_angstroms)
      ori_samp = np.arange(-180 / arg.nfold, 180 / arg.nfold - 0.001,
                           arg.grid_resolution_ori_degrees)
      sampler = rp.sampling.grid_sym_axis(crt_smap, ori_samp, axis=[0, 0, 1], flip=[0, 1, 0])
      logging.info(f'docking samples per splice {len(sampler)}')
   elif arg.docking_method.lower() == 'hier':
      search = rp.hier_search
      sampler = rp.sampling.hier_axis_sampler(arg.nfold, lb=cb[0], ub=cb[1])
      logging.info(f'docking possible samples per splice {sampler.size(4)}')
   else:
      raise ValueError(f'unknown search dock_method {arg.dock_method}')

   enddir = dict(N='C', C='N')[criteria.bbspec[-1].direction[0]]
   results = list()
   futures = list()
   exe = worms.util.InProcessExecutor()
   if arg.parallel > 1:
      exe = concurrent.futures.ProcessPoolExecutor(arg.parallel)

   logging.info(f'docking {min(max_dock, len(wresult.idx))} of {len(wresult.idx)} worms fusions')
   with exe as pool:
      for iresult, (idx, pos) in enumerate(zip(wresult.idx, wresult.pos)):
         if max_dock > 0 and iresult >= max_dock:
            break
         pose, prov = worms.make_pose_crit(arg.db[0], ssdag, criteria, idx, pos, provenance=True)
         arg.prof.checkpoint('make pose')
         label, bbnames = make_label(ssdag, idx, **arg)
         futures.append(
            pool.submit(plug_dock_one, hole, search, sampler, pose, prov, label, bbnames, enddir,
                        iresult, **arg.sub(database=None, dont_store_body_in_results=True)))
      results, plugs, profs = zip(*[f.result() for f in futures])
   arg.prof.merge(profs)

   results = rp.search.concat_results(results)
   if not arg.dont_store_body_in_results:
      results.body_labels = 'plug hole'.split()
      results.bodies = [(p, hole) for p in plugs]

   return results

def plug_dock_one(hole, search, sampler, pose, prov, label, bbnames, enddir, iresult, **kw):
   arg = Bunch(kw)
   prof = rp.Timer().start()
   plug = rp.body.Body(pose, which_ss="H", trim_direction=enddir, label=label, components=bbnames)
   prof.checkpoint('make body')
   op = f'{arg.output_prefix}_{iresult:04}'
   # read from forked cache data
   hscore = cache(rp.RpxHier, arg.hscore_files, hscore_data_dir=arg.hscore_data_dir)
   result = make_plugs(plug, hole, hscore, search, sampler, **arg.sub(output_prefix=op))
   prof.checkpoint('plug dock')
   if arg.dont_store_plugs:
      plug = None
   result.pdb_extra = [get_pdb_extra(pose, prov)] * len(result.data.scores)
   return result, plug, prof

def get_pdb_extra(pose, prov):
   mod, new, lost, junct = get_affected_positions(pose, prov)
   commas = lambda l: ",".join(str(_) for _ in l)
   extra = ''
   for ip, p in enumerate(prov):
      lb, ub, psrc, lbsrc, ubsrc = p
      extra += (f"Segment: {ip:2} resis {lb:4}-{ub:4} come from resis " +
                f"{lbsrc}-{ubsrc} of {psrc.pdb_info().name()}\n")
   nchain = pose.num_chains()
   extra += "Modified positions: " + commas(mod) + "\n"
   extra += "New contact positions: " + commas(new) + "\n"
   extra += "Lost contact positions: " + commas(lost) + "\n"
   extra += "Junction residues: " + commas(junct) + "\n"
   extra += "Length of asymetric unit: " + str(len(pose.residues)) + "\n"
   extra += "Number of chains in ASU: " + str(nchain) + "\n"
   return extra

def make_label(ssdag, idx, sep='__', **kw):
   bbs = [ssdag.bbs[i][ssdag.verts[i].ibblock[idx[i]]] for i in range(len(idx))]
   names = [bytes(b.name).decode() for b in bbs]
   splice = [ssdag.verts[i].ires[idx[i]] for i in range(len(idx))]
   splice = np.concatenate(splice)
   splice = np.array([s for s in splice if s >= 0])
   splice = splice.reshape(-1, 2)
   ndigit = rp.util.num_digits(max(np.max(v.ires) for v in ssdag.verts))
   s = str()
   for i, n in enumerate(names):
      if i > 0:
         s += f'_r{splice[i-1,0]:0{ndigit}}{sep}r{splice[i-1,1]:0{ndigit}}_'
      s += n
   return f'_{s}_', names

def main_test_dump():
   arg, crit = setup()
   result = rp.load('worms_Result.pickle')
   assert isinstance(result, rp.Result)
   arg.dump_pbds = True
   result.dump_pdbs_top_score(**arg)
   result.dump_pdbs_top_score_each(**arg)

if __name__ == '__main__':
   # main_test_dump()
   main()
