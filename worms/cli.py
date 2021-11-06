import sys, argparse, collections, logging

import worms
from worms import Bunch, PING

from worms.criteria import *

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
   only_outputs=[-1],
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
   splice_min_dotz=0.0,
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
   max_score0sym=9e9,
   max_porosity=9e9,
   max_com_redundancy=1.0,
   full_score0sym=0,
   #
   output_from_pose=1,
   output_symmetric=1,
   output_prefix="./worms",
   output_suffix="",
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

   # define which helices are "horizontal"
   # =====================================
   helixconf_min_num_horiz_helix=0,  # min number of 'horizontal helices required'
   helixconf_max_num_horiz_helix=999,  # max number of 'horizontal helices required'
   helixconf_helix_res_to_ignore=[int],  # ignore helices in specified residue range(s)
   helixconf_min_vert_angle=0.0,  # min angle out of horizontal (perpendicular) plane
   helixconf_max_vert_angle=8.0,  # max angle out of horizontal (perpendicular) plane
   helixconf_min_horiz_angle=0.0,  # min horizontal 'twist' around sym axis
   helixconf_max_horiz_angle=999.0,  # max horizontal 'twist' around sym axis
   helixconf_min_dist_to_sym_axis=0.0,  # min distance from helix to sym axis
   helixconf_max_dist_to_sym_axis=999.0,  # max distance from helix to sym axis
   helixconf_check_helix_curve=0,  # consider helix curviture
   helixconf_strict_helix_curve=0,  # all regions of helix curve must angles csts
   helixconf_lax_helix_curve=0,  # single region of helix curve must satisfy angle csts
   helixconf_min_helix_size=14,
   helixconf_max_helix_size=9999,

   # define which helices are on surface / extrema
   # =============================================
   helixconf_use_hull_for_surface=1,  # use a convex hull to define extrema
   helixconf_max_num_hull_verts=9999,  # limit number of points in null
   helixconf_hull_ignore_loops=0,  # ignore loops in hull
   helixconf_hull_trim_extrema_dist=0,  # trim points on extrema close to hull and recompute
   helixconf_hull_trim_extrema_max_snug_dist=0,  # trim points until all hull is 'snug' to within dist
   helixconf_max_res_trimed_extrema=0,  # max res allowed to be trimmed from hull
   helixconf_min_depth_within_hull=0.0,  # furthist helix burial within hull allowed
   helixconf_max_depth_within_hull=2.0,  # furthist helix exposure outside hull allowed
   helixconf_min_axis_dist_trim=0.0,  # dont trim res within dist of symaxis
   helixconf_max_axis_dist_trim=0.0,  # dont trim res beyond dist of symaxis
   helixconf_local_extrema_distance=9e9,

   # define multiple helix patches
   # =============================
   helixconf_min_neighbor_helix=0,  # minimum number of neighboring helices
   helixconf_max_neighbor_helix=999,  # minimum number of neighboring helices   
   helixconf_min_abs_depth_neighbor=0,  # furthist neighbor burial within hull
   helixconf_max_abs_angle_neighbor=2,  # furthist neighbor exposuer outside hull
   helixconf_min_rel_depth_among_neighbor_helix=0,  # max neighbor vertical separation
   helixconf_max_rel_angle_to_neighbor_helix=0,  # max neighbor helix angle delta
   helixconf_min_neighbor_res_contacts=0,  # look at res pairs to determine neighbor distance
   helixconf_max_dist_to_be_neighbor_helix=0,  # neighbor must be within dist of primary
   helixconf_min_neighbor_vert_dist=0,  # min neighbor separation along sym axis
   helixconf_max_neighbor_vert_dist=0,  # max neighbor separation along sym axis
   helixconf_min_neighbor_horizontal_dist=0,  # min neighbor separation perp to sym axis
   helixconf_max_neighbor_horizontal_dist=0,  # max neighbor separation perp to sym axis
   helixconf_both_ends_neighbor_helix=0,  # neighbor helix contacts must span helices
   #
   shuffle_output=0,
   xtal_min_cell_size=0,
   xtal_max_cell_size=9e9,
   #
   print_splice_fail_summary=1,
   print_info_edges_with_no_splices=1,
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
   parser = parent if parent else argparse.ArgumentParser(allow_abbrev=False)

   for k, v in cli_args.items():
      nargs = None
      type_ = type(v)
      if isinstance(v, list):
         nargs = "+"
         if isinstance(v[0], type):
            type_ = v[0]
            v = list()
         else:
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
   if argv is None:
      argv = sys.argv[1:]
   for a in argv.copy():
      if not a.startswith('@'):
         continue
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
   return worms.Bunch(vars(arg))

BBDir = collections.namedtuple('BBDir', ('bblockspec', 'direction'))

def _bbspec(bb, nc):
   return list(BBDir(*x) for x in zip(bb, nc))

def build_worms_setup_from_cli_args(
   argv=None,
   parser=None,
   construct_databases=True,
):

   if argv is None: argv = sys.argv[1:]

   kw = get_cli_args(argv, parser)

   numeric_level = getattr(logging, kw.loglevel.upper(), None)
   if not isinstance(numeric_level, int):
      raise ValueError('Invalid log level: %s' % kw.loglevel)
   logging.getLogger().setLevel(numeric_level)

   if kw.config_file == [""]:
      kw.config_file = []
   kw.topology = worms.topology.Topology(kw.topology)
   if not kw.config_file:
      if not kw.geometry or not kw.geometry[0]:
         print('--geometry not specified')
         sys.exit()
      crit = eval("".join(kw.geometry))
      bb = kw.bbconn[1::2]
      nc = kw.bbconn[0::2]
      kw.topology.check_nc(nc)
      crit.bbspec = _bbspec(bb, nc)
      assert len(nc) == len(bb)
      assert crit.from_seg < len(bb)
      assert crit.to_seg < len(bb)
      if isinstance(crit, Cyclic) and crit.origin_seg is not None:
         assert crit.origin_seg < len(bb)
      crit = [crit]

   else:
      crit = []
      for cfile in kw.config_file:
         with open(cfile) as inp:
            lines = inp.readlines()
            assert len(lines) == 2

            def orient(a, b):
               return (a or "_") + (b or "_")

            bbnc = eval(lines[0])
            bb = [x[0] for x in bbnc]
            nc = [x[1] for x in bbnc]
            kw.topology.check_nc(nc)

            crit0 = eval(lines[1])
            crit0.bbspec = _bbspec(bb, nc)
            assert len(nc) == len(bb)
            assert crit0.from_seg < len(bb)
            assert crit0.to_seg < len(bb)
            if isinstance(crit0, Cyclic) and crit0.origin_seg is not None:
               assert crit0.origin_seg < len(bb)
            crit.append(crit0)

   # TODO oh god... fix these huge assumptions about Criteria
   for c in crit:
      # c.tolerance = kw.tolerance
      c.lever = kw.lever
      c.rot_tol = c.tolerance / kw.lever

   if kw.max_score0 > 9e8:
      kw.max_score0 = 2.0 * len(crit[0].bbspec)

   if kw.merge_bblock < 0:
      kw.merge_bblock = None
   if kw.only_merge_bblocks == [-1]:
      kw.only_merge_bblocks = []
   if kw.only_bblocks == [-1]:
      kw.only_bblocks = []
   if kw.only_ivertex == [-1]:
      kw.only_ivertex = []
   if kw.only_outputs == [-1]:
      kw.only_outputs = []
   if kw.bblock_ranges == [-1]:
      kw.bblock_ranges = []
   elif kw.shuffle_bblocks:
      print("you probably shouldnt use --shuffle_bblocks with --bblock_ranges ")
      sys.exit(0)
   if kw.merge_segment == -1:
      kw.merge_segment = None
   kw.tolerance = min(kw.tolerance, 9e8)

   if kw.dbfiles == [""]:
      assert 0, "no --dbfiles specified"

   if len(kw.nbblocks) == 1:
      kw.nbblocks *= 100
   if kw.output_only_connected != 'auto':
      if kw.output_only_connected in ('', 0, '0', 'false', 'False'):
         kw.output_only_connected = False
      else:
         kw.output_only_connected = True

   if construct_databases:
      if kw.disable_cache:
         kw.database = worms.database.Databases(
            worms.database.BBlockDB(**kw),
            worms.database.SpliceDB(**kw),
         )
      else:
         kw.database = worms.database.Databases(
            worms.database.CachingBBlockDB(**kw),
            worms.database.CachingSpliceDB(**kw),
         )
      kw.db = kw.database  # depricated

   # print("-------------- arg ---------------")
   # for k, v in kw.items():
   #    print("   ", k, v)
   # print("-----------------------------------")

   return crit, kw
