import os

if __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import willutil as wu

import worms, worms.viz  # type: ignore
from worms.cli import BBDir

def main():

   timer = wu.Timer()

   # argv = ['@' + worms.data.test_file_path(f'{testname}/config/{testname}.flags')]
   # criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)

   # import worms.bblock
   # import numba
   # print(numba.experimental.jitclass)
   # assert 0
   kw = getopts_test_ext()
   timer.checkpoint('getopts_test_ext')

   kw.dbfiles = [
      '/home/yhsia/helixfuse/2018-07-09_sym/processing/database/HFuse_Cx_database.20180711.txt',
      # '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.txt',
      '/home/sheffler/debug/worms_extension/input/straight_DHR.local.json',
   ]
   kw.min_seg_len = 200
   kw.nbblocks = 2

   # TODO remove me
   import sys
   if len(sys.argv) > 1:
      kw.nbblocks = int(sys.argv[1])
      print(kw.nbblocks)

   kw.tolerance = 2.0
   kw.max_score0 = 10
   kw.max_score0sym = 50
   kw.precache_splices = False
   # kw.only_bblocks = [
   # [8, 10,11,12,13],
   # [11,14,15],
   # [3],
   # ]

   timer.checkpoint('make db')
   bbspec = [
      BBDir(bblockspec='C3_C', direction='_C'),
      # BBDir(bblockspec='Het:CN', direction='NC'),
      BBDir(bblockspec='straight_DHR', direction='NC'),
      BBDir(bblockspec='C2_N', direction='N_'),
   ]

   # criteria = worms.criteria.NullCriteria(bbspec=bbspec)
   criteria = worms.criteria.Octahedral(c3=0, c2=-1, bbspec=bbspec)
   kw.database = worms.database.Databases(
      worms.database.CachingBBlockDB(**kw),
      worms.database.CachingSpliceDB(**kw),
   )
   print('built database', flush=True)

   if os.path.exists('test_extension.pickle'):
      result = wu.load('test_extension.pickle')
   else:

      ssdag = worms.ssdag.simple_search_dag(criteria, lbl='all', **kw).ssdag
      print('built ssdag', flush=True)

      result = worms.search.grow_linear(
         ssdag=ssdag,
         loss_function=criteria.jit_lossfunc(**kw),
         last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
         lbl='alltogether',
         debug=False,
         **kw,
      )
      print('finish search', flush=True)
      result = worms.search.result.ResultTable(result, ssdag)
      if len(result.idx) > 0:
         wu.save(result, 'test_extension.pickle')
      print('saved results', flush=True)

   timer.checkpoint('make/load results')

   ssdag = result.ssdag
   print('nresults', len(result.idx))

   worms.app.simple.output_simple(criteria, ssdag, result, output_suffix='',
                                  **kw.sub(output_prefix='./foo'))
   timer.checkpoint('output_simple')

   std_output = False

   if std_output:
      result.add('zheight', np.zeros_like(result.err))
      result.add('zradius', np.zeros_like(result.err))
      result.add('radius', np.zeros_like(result.err))
      result.add('porosity', -np.ones_like(result.err))

      print('&&&&&&&&&&&&&&&&&&&&&& filter_and_output_results &&&&&&&&&&&&&&&&&&&&')
      worms.output.filter_and_output_results(criteria, ssdag, result, **kw)
      print('&' * 80)
      # print(ssdag.bblocks)
      # print(ssdag.bblocks[0][0].connections)
      timer.checkpoint('output std')
   # assert 0, 'done std_output'

   #

   # iresult = 1

   # for iseg, v in enumerate(ssdag.verts):
   #    ivert = result.idx[iresult, iseg]
   #    ibb = v.ibblock[ivert]
   #    bb = ssdag.bblocks[iseg][ibb]
   #    wu.showme(bb, bounds=(0, -1), name=f'foo{iseg}', pos=np.eye(4))
   result.criteria = criteria
   assert isinstance(result, worms.Result)
   assert not isinstance(result, np.ndarray)
   # print(type(result))
   # wu.showme(result, iresult=iresult, headless=False, showextras=True, name='worms_result_aysm',
   #           sym=wu.sym.frames('oct')[:1])
   # wu.showme(result, iresult=iresult, headless=False, showextras=True, name='worms_result_sym',
   #           sym=wu.sym.frames('oct')[1:])

   print('test_extension done')

   print(timer)

def getopts_test_ext():
   kw = wu.Bunch(
      monte_carlo=0,
      no_duplicate_bases=0,
      max_output=100,
      shuffle_bblocks=0,
      parallel=0,
      # helixconf_min_num_horiz_helix=2,
      helixconf_min_num_horiz_helix=0,
      helixconf_max_vert_angle=3,
      helixconf_max_depth_within_hull=2,
      helixconf_use_hull_for_surface=0,
      helixconf_min_helix_size=14,
      disable_cache=0,
      pbar=0,
      cachedirs='./testcache',
      lever=25,
      splice_rms_range=4,
      # splice_max_rms=0.8,
      splice_max_rms=2,
      splice_clash_contact_by_helix=1,
      splice_ncontact_cut=1,
      splice_ncontact_no_helix_cut=1,
      splice_nhelix_contacted_cut=1,
      splice_max_chain_length=9999,
      output_from_pose=True,
      merge_bblock=-1,
      output_symmetric=True,
      output_centroid=True,
      output_prefix='test_extension_',
      output_only_AAAA=False,
      full_score0sym=False,
      output_short_fnames=True,
      output_only_connected='auto',
      null_base_names=["", "?", "n/a", "none"],
      only_outputs=[],
      postfilt_splice_max_rms=10,
      postfilt_splice_rms_length=4,
      postfilt_splice_ncontact_cut=0,
      postfilt_splice_ncontact_no_helix_cut=0,
      postfilt_splice_nhelix_contacted_cut=0,
      merge_segment=-1,
   )
   return kw

if __name__ == '__main__':
   main()
'''
/home/yhsia/helixfuse/2018-07-09_sym/processing/database/HFuse_Cx_database.20180711.txt
/home/yhsia/helixfuse/2018-07-09_sym/processing/database/HFuse_Cx_database.20180914.txt
/home/yhsia/helixfuse/2019-04-16_sym_r2/processing/database/HFuse_Cx_database.20190422.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180406.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180406.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh3e_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh3e_3.20180406.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180406.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180406.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh3e_3.20180213.txt
/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh3e_3.20180406.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-2.8.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-2.9.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.0.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.1.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.2.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.3.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.4.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20190218.FILTERED.-3.5.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180412.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180416.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180516.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20190218.FILTERED.-3.3.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20190218.FILTERED.-3.4.txt
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20190218.FILTERED.-3.5.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-131_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180516.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180321.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180411.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180412.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180416.txt
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180516.txt
/home/yhsia/helixfuse/cyc_george/processing/database/HFuse_Gcyc_database.20180702.txt
/home/yhsia/helixfuse/cyc_george/processing/database/HFuse_Gcyc_database.20180817.txt
/home/yhsia/helixfuse/rosetta_scripts_ver/processing/database/HFuse_Cx_database.20180217.txt
/home/yhsia/helixfuse/rosetta_scripts_ver/processing/database/HFuse_Cx_database.20180219.txt
/home/yhsia/helixfuse/sym_r2/processing/database/HFuse_Cx_database.20180226.txt
/home/yhsia/helixfuse/sym_r2/processing/database/HFuse_Cx_database.20180325.txt
/home/yhsia/helixfuse/sym_r2/processing/database/HFuse_Cx_database.20180411.txt
/home/yhsia/helixfuse/sym_r2/processing/database/HFuse_Cx_database.20180416.txt
'''