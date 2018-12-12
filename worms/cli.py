import sys
import argparse

from worms import util
from worms.criteria import *
from worms.topology import Topology
from worms.database import CachingBBlockDB, CachingSpliceDB
from worms.database import NoCacheBBlockDB, NoCacheSpliceDB


def get_cli_args(argv=None, **kw):
    """
    lazy definition of cli args via a dictionary (kw) mapping names to
    default values
    """
    if argv is None: argv = sys.argv[1:]
    # add from @files
    atfiles = []
    for a in argv:
        if a.startswith('@'):
            atfiles.append(a)
    for a in atfiles:
        argv.remove(a)
        with open(a[1:]) as inp:
            newargs = []
            for l in inp:
                # last char in l is newline, so [:-1] ok
                newargs.extend(l[:l.find('#')].split())
            argv = newargs + argv

    p = argparse.ArgumentParser()
    for k, v in kw.items():
        nargs = None
        type_ = type(v)
        if isinstance(v, list):
            nargs = '+'
            type_ = type(v[0])
        p.add_argument('--' + k, type=type_, dest=k, default=v, nargs=nargs)
        # print('arg', k, type_, nargs, v)
    args = p.parse_args(argv)
    if hasattr(args, 'parallel') and args.parallel < 0:
        args.parallel = util.cpu_count()
    return args


def build_worms_setup_from_cli_args(argv):
    args = get_cli_args(
        argv=argv,
        geometry=[''],
        bbconn=[''],
        config_file=[''],
        nbblocks=64,
        use_saved_bblocks=0,
        monte_carlo=[0.0],
        parallel=1,
        verbosity=2,
        precache_splices=1,
        precache_splices_and_quit=0,
        pbar=0,
        pbar_interval=10.0,
        #
        context_structure='',
        #
        cachedirs=[''],
        disable_cache=0,
        dbfiles=[''],
        load_poses=0,
        read_new_pdbs=0,
        run_cache='',
        merge_bblock=-1,
        no_duplicate_bases=1,
        shuffle_bblocks=1,
        only_merge_bblocks=[-1],
        only_bblocks=[-1],
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
        max_output=1000,
        max_score0=9e9,
        max_porosity=1.0,
        #
        output_from_pose=1,
        output_symmetric=1,
        output_prefix='./worms',
        output_centroid=0,
        output_only_AAAA=0,
        #
        cache_sync=0.003,
        #
        postfilt_splice_max_rms=0.7,
        postfilt_splice_rms_length=9,
        postfilt_splice_ncontact_cut=40,
        postfilt_splice_ncontact_no_helix_cut=2,
        postfilt_splice_nhelix_contacted_cut=3,

    )
    if args.config_file == ['']:
        args.config_file = []
    args.topology = Topology(args.topology)
    if not args.config_file:
        crit = eval(''.join(args.geometry))
        bb = args.bbconn[1::2]
        nc = args.bbconn[0::2]
        args.topology.check_nc(nc)
        crit.bbspec = list(list(x) for x in zip(bb, nc))
        assert len(nc) == len(bb)
        assert crit.from_seg < len(bb)
        assert crit.to_seg < len(bb)
        if isinstance(crit, Cyclic) and crit.origin_seg is not None:
            assert crit.origin_seg < len(bb)
        crit = [crit]
    else:
        crit = []
        for cfile in args.config_file:
            with open(cfile) as inp:
                lines = inp.readlines()
                assert len(lines) is 2

                def orient(a, b):
                    return (a or '_') + (b or '_')

                bbnc = eval(lines[0])
                bb = [x[0] for x in bbnc]
                nc = [x[1] for x in bbnc]
                args.topology.check_nc(nc)

                crit0 = eval(lines[1])
                crit0.bbspec = list(list(x) for x in zip(bb, nc))
                assert len(nc) == len(bb)
                assert crit0.from_seg < len(bb)
                assert crit0.to_seg < len(bb)
                if isinstance(crit0, Cyclic) and crit0.origin_seg is not None:
                    assert crit0.origin_seg < len(bb)
                crit.append(crit0)

    # oh god... fix these huge assumptions about Criteria
    for c in crit:
        # c.tolerance = args.tolerance
        c.lever = args.lever
        c.rot_tol = c.tolerance / args.lever

    if args.max_score0 > 9e8:
        args.max_score0 = 2.0 * len(crit[0].bbspec)

    if args.merge_bblock < 0: args.merge_bblock = None
    if args.only_merge_bblocks == [-1]:
        args.only_merge_bblocks = []
    if args.only_bblocks == [-1]:
        args.only_bblocks = []
    if args.merge_segment == -1:
        args.merge_segment = None

    if args.dbfiles == ['']:
        assert 0, 'no --dbfiles specified'

    kw = vars(args)
    if args.disable_cache:
        kw['db'] = NoCacheBBlockDB(**kw), NoCacheSpliceDB(**kw)
    else:
        kw['db'] = CachingBBlockDB(**kw), CachingSpliceDB(**kw)

    print('-------------- args ---------------')
    for k, v in kw.items():
        print('   ', k, v)
    print('-----------------------------------')

    kw['db'][0].report()

    return crit, kw
