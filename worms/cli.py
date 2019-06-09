import sys, argparse, collections, logging

from worms import util
from worms.criteria import *
from worms.topology import Topology
from worms.database import CachingBBlockDB, CachingSpliceDB
from worms.database import NoCacheBBlockDB, NoCacheSpliceDB

cli_args = dict(
   loglevel='INFO',
   geometry=[""],
   bbconn=[""],
   config_file=[""],
   null_base_names=["", "?", "n/a", "none"],
   nbblocks=[64],
   use_saved_bblocks=0,
   monte_carlo=[0.0],
   parallel=1,
   verbosity=2,
   precache_splices=1,
   precache_splices_and_quit=0,
   pbar=0,
   pbar_interval=10.0,
   #
   context_structure="",
   #
   cachedirs=[""],
   disable_cache=0,
   dbfiles=[""],
   dbroot="",
   load_poses=0,
   read_new_pdbs=0,
   run_cache="",
   merge_bblock=-1,
   no_duplicate_bases=1,
   shuffle_bblocks=1,
   only_merge_bblocks=[-1],
   only_bblocks=[-1],  # select single set of bbs
   only_ivertex=[-1],  # only for debugging
   bblock_ranges=[-1],
   merge_segment=-1,
   min_seg_len=15,
   topology=[-1],
   # splice stuff
   splice_rms_range=4,
   splice_max_rms=0.7,
   splice_clash_d2=3.5**2,  # ca only
   splice_contact_d2=8.0**2,
   splice_clash_contact_range=40,
   splice_clash_contact_by_helix=1,
   splice_ncontact_cut=38,
   splice_ncontact_no_helix_cut=6,
   splice_nhelix_contacted_cut=3,
   splice_max_chain_length=450,
   #
   tolerance=1.0,
   lever=25.0,
   min_radius=0.0,
   hash_cart_resl=1.0,
   hash_ori_resl=5.0,
   loose_hash_cart_resl=10.0,
   loose_hash_ori_resl=20.0,
   merged_err_cut=999.0,
   rms_err_cut=3.0,
   ca_clash_dis=3.0,
   disable_clash_check=0,
   #
   max_linear=1000000,
   max_merge=100000,
   max_clash_check=10000,
   max_dock=100000,
   max_output=1000,
   max_score0=9e9,
   max_porosity=9e9,
   max_com_redundancy=1.0,
   full_score0sym=0,
   #
   output_from_pose=1,
   output_symmetric=1,
   output_prefix="./worms",
   output_centroid=0,
   output_only_AAAA=0,
   output_short_fnames=0,
   output_only_connected='auto',
   #
   cache_sync=0.003,
   #
   postfilt_splice_max_rms=0.7,
   postfilt_splice_rms_length=9,
   postfilt_splice_ncontact_cut=40,
   postfilt_splice_ncontact_no_helix_cut=2,
   postfilt_splice_nhelix_contacted_cut=3,
)

def add_argument_unless_exists(parser, *arg, **kw):
   try:
      parser.add_argument(*arg, **kw)
   except argparse.ArgumentError:
      pass

def make_cli_arg_parser(parent=None):
   """
    lazy definition of cli arg via a dictionary (kw) mapping names to
    default values
    """
   parser = parent if parent else argparse.ArgumentParser()

   for k, v in cli_args.items():
      nargs = None
      type_ = type(v)
      if isinstance(v, list):
         nargs = "+"
         type_ = type(v[0])

      add_argument_unless_exists(
         parser,
         "--" + k,
         type=type_,
         dest=k,
         default=v,
         nargs=nargs,
      )
      # print('arg', k, type_, nargs, v)
   parser._has_worms_args = True
   return parser

def make_argv_with_atfiles(argv=None):
   if argv is None: argv = sys.argv[1:]
   for a in argv.copy():
      if not a.startswith('@'): continue
      argv.remove(a)
      with open(a[1:]) as inp:
         newargs = []
         for l in inp:
            # last char in l is newline, so [:-1] ok
            newargs.extend(l[:l.find("#")].split())
         argv = newargs + argv
   return argv

def get_cli_args(argv=None, parser_=None):
   if not parser_ or not hasattr(parser_, '_has_worms_args'):
      parser_ = make_cli_arg_parser(parser_)
   argv = make_argv_with_atfiles(argv)
   arg = parser_.parse_args(argv)
   if hasattr(arg, "parallel") and arg.parallel < 0:
      arg.parallel = util.cpu_count()
   return arg

BBDir = collections.namedtuple('BBDir', ('bblockspec', 'direction'))

def _bbspec(bb, nc):
   return list(BBDir(*x) for x in zip(bb, nc))

def build_worms_setup_from_cli_args(argv, parser=None):
   arg = get_cli_args(argv, parser)

   numeric_level = getattr(logging, arg.loglevel.upper(), None)
   if not isinstance(numeric_level, int):
      raise ValueError('Invalid log level: %s' % arg.loglevel)
   logging.getLogger().setLevel(numeric_level)

   if arg.config_file == [""]: arg.config_file = []
   arg.topology = Topology(arg.topology)
   if not arg.config_file:
      crit = eval("".join(arg.geometry))
      bb = arg.bbconn[1::2]
      nc = arg.bbconn[0::2]
      arg.topology.check_nc(nc)
      crit.bbspec = _bbspec(bb, nc)
      assert len(nc) == len(bb)
      assert crit.from_seg < len(bb)
      assert crit.to_seg < len(bb)
      if isinstance(crit, Cyclic) and crit.origin_seg is not None:
         assert crit.origin_seg < len(bb)
      crit = [crit]
   else:
      crit = []
      for cfile in arg.config_file:
         with open(cfile) as inp:
            lines = inp.readlines()
            assert len(lines) is 2

            def orient(a, b):
               return (a or "_") + (b or "_")

            bbnc = eval(lines[0])
            bb = [x[0] for x in bbnc]
            nc = [x[1] for x in bbnc]
            arg.topology.check_nc(nc)

            crit0 = eval(lines[1])
            crit0.bbspec = _bbspec(bb, nc)
            assert len(nc) == len(bb)
            assert crit0.from_seg < len(bb)
            assert crit0.to_seg < len(bb)
            if isinstance(crit0, Cyclic) and crit0.origin_seg is not None:
               assert crit0.origin_seg < len(bb)
            crit.append(crit0)

   # oh god... fix these huge assumptions about Criteria
   for c in crit:
      # c.tolerance = arg.tolerance
      c.lever = arg.lever
      c.rot_tol = c.tolerance / arg.lever

   if arg.max_score0 > 9e8:
      arg.max_score0 = 2.0 * len(crit[0].bbspec)

   if arg.merge_bblock < 0: arg.merge_bblock = None
   if arg.only_merge_bblocks == [-1]: arg.only_merge_bblocks = []
   if arg.only_bblocks == [-1]: arg.only_bblocks = []
   if arg.only_ivertex == [-1]: arg.only_ivertex = []
   if arg.bblock_ranges == [-1]: arg.bblock_ranges = []
   elif arg.shuffle_bblocks:
      print("you probably shouldnt use --shuffle_bblocks with --bblock_ranges ")
      sys.exit(0)
   if arg.merge_segment == -1: arg.merge_segment = None
   arg.tolerance = min(arg.tolerance, 9e8)

   if arg.dbfiles == [""]:
      assert 0, "no --dbfiles specified"

   if len(arg.nbblocks) == 1: arg.nbblocks *= 100
   if arg.output_only_connected != 'auto':
      if arg.output_only_connected in ('', 0, '0', 'false', 'False'):
         arg.output_only_connected = False
      else:
         arg.output_only_connected = True

   kw = vars(arg)
   if arg.disable_cache: kw["db"] = NoCacheBBlockDB(**kw), NoCacheSpliceDB(**kw)
   else: kw["db"] = CachingBBlockDB(**kw), CachingSpliceDB(**kw)

   print("-------------- arg ---------------")
   for k, v in kw.items():
      print("   ", k, v)
   print("-----------------------------------")

   kw["db"][0].report()

   return crit, kw
