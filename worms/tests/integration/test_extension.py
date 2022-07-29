DISABLE_NUMBA = True
CREATE_NEW_RESULTS = False

if DISABLE_NUMBA:
   # if __name__ == '__main__':
   import os
   os.environ['NUMBA_DISABLE_JIT'] = '1'
   print()
   print()
   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   print('NUMBA_DISABLE_JIT')
   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   print()
   print()

import worms
import worms.rosetta_init
import willutil as wu

def main():
   # _test_extension()
   test_extension_output()

def test_extension_output(sym='oct'):

   argv = ['@' + worms.data.test_file_path(f'test_extension/config/test_extension_{sym}.flags')]
   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)
   kw.timer = wu.Timer()
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   #dbfiles = [
   #   '/home/sheffler/src/worms/worms/data/test_cases/test_extension/databases/test_extension__mbb0000__minimal_replicate_database.txz'
   #]

   # bbdb = worms.database.BBlockDB(
   #    # cachedirs='test_extension_cache',
   #    lazy=False,
   #    **kw.sub(read_new_pdbs=True),
   # )
   kw.timer.checkpoint('startup')
   tmpfn = f'test_results_{sym}.pickle'

   if CREATE_NEW_RESULTS:
      ssdag, result = worms.app.run_simple(criteria, **kw)
      result2 = worms.filters.prune_clashes(ssdag, criteria, result, **kw)
      kw.timer.checkpoint('prune_clashes')
      print('clashes:', len(result.idx), len(result2.idx))
      result = result2
      assert len(result) > 0
      kw.timer.checkpoint('run simple')
      wu.save((ssdag, result), tmpfn)
      kw.timer.checkpoint(f'save {tmpfn}')
      # assert 0
   else:
      ssdag, result = wu.load(tmpfn)
      kw.timer.checkpoint('load test results')

      # assert 0

      # import pickle
      # with open(
      #       '/home/sheffler/src/worms/worms/data/test_cases/test_extension/testcache/test_extension_reference_results.pickle',
      #       'rb') as inp:
   #    crit, ssdag, result = pickle.load(inp)
   iresult = 0

   extensions = {1: 3}

   if True:
      sinfo = ssdag.get_structure_info(result.idx[0])
      # print(sinfo)

      xalign = criteria.alignment(result.pos[iresult])
      xalign, extpos = worms.output.dumppdb.modify_xalign_by_extension(
         ssdag,
         result.idx[iresult],
         result.pos[iresult],
         xalign,
         sinfo.bblocks,
         kw.database.bblockdb,
         extensions=extensions,
         **kw,
      )
      # # modify alignment
      pose = worms.ssdag.make_pose_simple(
         ssdag,
         result.idx[iresult],
         xalign @ extpos,
         extensions=extensions,
         only_spliced_regions=True,
         **kw,
      )
      pose.dump_pdb('make_pose_simple.pdb')
      assert 0

   worms.app.output_simple(
      criteria,
      ssdag,
      result,
      pos=result.pos[iresult],
      sym=criteria.symname,
      # extensions=extensions,
      output_indices=[0],
      **kw,
   )
   kw.timer.checkpoint('output_simple')

   kw.timer.report()
   print("TEST EXTENSION DONE")

   # assert 0

def _test_extension():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   worms.tests.generic_integration_test('test_extension')

if __name__ == '__main__':
   main()
