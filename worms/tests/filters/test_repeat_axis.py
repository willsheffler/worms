import os

# if __name__ == '__main__':
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import willutil as wu

import worms, worms.viz
from worms.util import get_props_from_url
from worms.cli import BBDir
import willutil as wu

def main():
   test_repeat_axis()

def test_repeat_axis():

   dbfiles = [
      'test_extension__mbb0000__minimal_replicate_database.txz',
      # 'test_extension__mbb0001__minimal_replicate_database.txz',
      # 'test_extension__mbb0002__minimal_replicate_database.txz',
      # 'test_extension__mbb0003__minimal_replicate_database.txz',
   ]
   kw = wu.Bunch(repeat_twist_tolerance=1.0)
   bbdb = worms.database.BBlockDB(
      cachedirs='./test_repeat_axis_cache',
      dbfiles=dbfiles,
      lazy=False,
      read_new_pdbs=True,
      **kw,
   )
   kw.database = wu.Bunch(bblockdb=bbdb)
   # _bblocks = bbdb.all_bblocks()
   _bblocks = bbdb.query('straight_DHR')
   bblocks = [worms.BBlock(bb, **kw) for bb in _bblocks]
   # print(len(bblocks), type(bblocks[0]))
   # print('built database', flush=True)
   shapes = [[(392, 3, 4), (731, 3, 4)], [(354, 3, 4), (507, 3, 4)], [(323, 3, 4), (650, 3, 4)]]
   starts = [[84, 183], [102, 153], [80, 189]]
   periods = [113, 51, 109]
   for ibb, (bblock, shape, start, period) in enumerate(zip(bblocks, shapes, starts, periods)):

      bblock3 = bblock.make_extended_bblock(nrepeats=3, **kw)
      assert get_props_from_url(bblock.pdbfile) == dict()
      assert get_props_from_url(bblock3.pdbfile) == dict(addrepeat=3)
      # print(bblock.ncac.shape == shape[0])
      # print(bblock3.ncac.shape == shape[1])
      assert bblock.repeatstart == start[0]
      assert bblock3.repeatstart == start[1]
      assert bblock.repeatspacing == period
      assert bblock3.repeatspacing == period

      # print(bblock.repeat_spacing[1])
      # print(bblock3.repeat_spacing[1])

      # wu.showme(bblock, bounds=(0, -1), name=f'bar{ibb}', pos=np.eye(4), showextras=True)
      # wu.showme(bblock3, bounds=(0, -1), name=f'ext2{ibb}', pos=np.eye(4), showextras=True)

   # argv = ['@' + worms.data.test_file_path(f'{testname}/config/{testname}.flags')]
   # criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)

   # import worms.bblock
   # import numba
   # print(numba.experimental.jitclass)
   # assert 0
   # kw = getopts_test_ext()
   # timer.checkpoint('getopts_test_ext')

   # ...

   # for ibb, bblock in enumerate(bblocks):
   # wu.showme(bblock, bounds=(0, -1), name=f'foo{ibb}', pos=np.eye(4), showextras=True)

   # if 0:  #os.path.exists('test_extension.pickle'):
   #    result = wu.load('test_extension.pickle')
   # else:

   #    ssdag = worms.ssdag.simple_search_dag(criteria, lbl='all', **kw).ssdag
   #    print('built ssdag', flush=True)

   #    result = worms.search.grow_linear(
   #       ssdag=ssdag,
   #       loss_function=criteria.jit_lossfunc(**kw),
   #       last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
   #       lbl='alltogether',
   #       debug=False,
   #       **kw,
   #    )
   #    print('finish search', flush=True)
   #    result = worms.search.result.ResultTable(result, ssdag)
   #    if len(result.idx) > 0:
   #       wu.save(result, 'test_extension.pickle')
   #    print('saved results', flush=True)
   # timer.checkpoint('make/load results')

   # ssdag = result.ssdag
   # print('nresults', len(result.idx))

   # worms.app.simple.output_simple(criteria, ssdag, result, output_suffix='',
   #                                **kw.sub(output_prefix='./foo'))
   # timer.checkpoint('output_simple')

   # std_output = False

   # if std_output:
   #    result.add('zheight', np.zeros_like(result.err))
   #    result.add('zradius', np.zeros_like(result.err))
   #    result.add('radius', np.zeros_like(result.err))
   #    result.add('porosity', -np.ones_like(result.err))

   #    print('&&&&&&&&&&&&&&&&&&&&&& filter_and_output_results &&&&&&&&&&&&&&&&&&&&')
   #    worms.output.filter_and_output_results(criteria, ssdag, result, **kw)
   #    print('&' * 80)
   #    # print(ssdag.bblocks)
   #    # print(ssdag.bblocks[0][0].connections)
   #    timer.checkpoint('output std')
   # # assert 0, 'done std_output'

   #

   # iresult = 0

   # # for iseg, v in enumerate(ssdag.verts):
   # #    ivert = result.idx[iresult, iseg]
   # #    ibb = v.ibblock[ivert]
   # #    bb = ssdag.bblocks[iseg][ibb]
   # #    wu.showme(bb, bounds=(0, -1), name=f'foo{iseg}', pos=np.eye(4))
   # result.criteria = criteria
   # assert isinstance(result, worms.Result)
   # assert not isinstance(result, np.ndarray)
   # # print(type(result))
   # # wu.showme(result, iresult=iresult, headless=False, showextras=True, name='worms_result_aysm',
   # # sym=wu.sym.frames('oct')[:1])
   # # wu.showme(result, iresult=iresult, headless=False, showextras=True, name='worms_result_sym',
   # #           sym=wu.sym.frames('oct')[1:])

   # print('test_extension done')

   # print(timer)

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
      splice_max_rms=1,
      splice_clash_contact_by_helix=1,
      splice_ncontact_cut=10,
      splice_ncontact_no_helix_cut=1,
      splice_nhelix_contacted_cut=2,
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
'''
loglevel
INFO
geometry
['Octahedral(c3=0,c2=-1)']
bbconn
['_C', 'C3_C', 'NC', 'straight_DHR', 'N_', 'C2_N']
config_file
[]
null_base_names
['', '?', 'n/a', 'none']
nbblocks
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
use_saved_bblocks
0
monte_carlo
[0.0]
parallel
0
verbosity
2
precache_splices
1
precache_splices_and_quit
0
pbar
0
pbar_interval
10.0
context_structure

cachedirs
['./testcache']
disable_cache
0
dbfiles
['test_extension__mbb0000__minimal_replicate_database.txz', 'test_extension__mbb0001__minimal_replicate_database.txz', 'test_extension__mbb0002__minimal_replicate_database.txz', 'test_extension__mbb0003__minimal_replicate_database.txz']
dbroot

load_poses
0
read_new_pdbs
0
run_cache

merge_bblock
None
no_duplicate_bases
0
shuffle_bblocks
0
only_merge_bblocks
[]
only_bblocks
[]
only_ivertex
[]
only_outputs
[]
bblock_ranges
[]
merge_segment
None
min_seg_len
200
topology
<worms.topology.Topology object at 0x7fe996140670>
splice_rms_range
4
splice_max_rms
1.0
splice_clash_d2
12.25
splice_contact_d2
64.0
splice_clash_contact_range
40
splice_clash_contact_by_helix
1
splice_ncontact_cut
10
splice_ncontact_no_helix_cut
1
splice_nhelix_contacted_cut
2
splice_max_chain_length
9999
splice_min_dotz
0.0
tolerance
1.0
lever
25.0
min_radius
0.0
hash_cart_resl
1.0
hash_ori_resl
5.0
loose_hash_cart_resl
10.0
loose_hash_ori_resl
20.0
merged_err_cut
999.0
rms_err_cut
3.0
ca_clash_dis
3.0
disable_clash_check
0
max_linear
1000000
max_merge
100000
max_clash_check
10000
max_dock
100000
max_output
100
max_score0
10.0
max_score0sym
50.0
max_porosity
9000000000.0
max_com_redundancy
1.0
full_score0sym
0
output_from_pose
1
output_symmetric
1
output_prefix
test_extension_
output_suffix

output_centroid
1
output_only_AAAA
0
output_short_fnames
0
output_only_connected
auto
cache_sync
0.003
postfilt_splice_max_rms
10.0
postfilt_splice_rms_length
4
postfilt_splice_ncontact_cut
0
postfilt_splice_ncontact_no_helix_cut
0
postfilt_splice_nhelix_contacted_cut
0
helixconf_min_num_horiz_helix
0
helixconf_max_num_horiz_helix
999
helixconf_helix_res_to_ignore
[]
helixconf_min_vert_angle
0.0
helixconf_max_vert_angle
3.0
helixconf_min_horiz_angle
0.0
helixconf_max_horiz_angle
999.0
helixconf_min_dist_to_sym_axis
0.0
helixconf_max_dist_to_sym_axis
999.0
helixconf_check_helix_curve
0
helixconf_strict_helix_curve
0
helixconf_lax_helix_curve
0
helixconf_min_helix_size
14
helixconf_max_helix_size
9999
helixconf_use_hull_for_surface
0
helixconf_max_num_hull_verts
9999
helixconf_hull_ignore_loops
0
helixconf_hull_trim_extrema_dist
0
helixconf_hull_trim_extrema_max_snug_dist
0
helixconf_max_res_trimed_extrema
0
helixconf_min_depth_within_hull
0.0
helixconf_max_depth_within_hull
2.0
helixconf_min_axis_dist_trim
0.0
helixconf_max_axis_dist_trim
0.0
helixconf_local_extrema_distance
9000000000.0
helixconf_min_neighbor_helix
0
helixconf_max_neighbor_helix
999
helixconf_min_abs_depth_neighbor
0
helixconf_max_abs_angle_neighbor
2
helixconf_min_rel_depth_among_neighbor_helix
0
helixconf_max_rel_angle_to_neighbor_helix
0
helixconf_min_neighbor_res_contacts
0
helixconf_max_dist_to_be_neighbor_helix
0
helixconf_min_neighbor_vert_dist
0
helixconf_max_neighbor_vert_dist
0
helixconf_min_neighbor_horizontal_dist
0
helixconf_max_neighbor_horizontal_dist
0
helixconf_both_ends_neighbor_helix
0
shuffle_output
0
xtal_min_cell_size
100
xtal_max_cell_size
9000000000.0
print_splice_fail_summary
1
print_info_edges_with_no_splices
1
save_minimal_replicate_database
0
repeat_axis_check
1
repeat_axis_weight
50.0
'''