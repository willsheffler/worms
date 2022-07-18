if __name__ == '__main__':
   import os
   os.environ['NUMBA_DISABLE_JIT'] = '1'
   print('NUMBA_DISABLE_JIT')

import worms
import willutil as wu

def main():
   # _test_extension()
   test_extension_output()

def test_extension_output():

   argv = ['@' + worms.data.test_file_path('test_extension/config/test_extension.flags')]
   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)
   kw.timer = wu.Timer()
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   #dbfiles = [
   #   '/home/sheffler/src/worms/worms/data/test_cases/test_extension/databases/test_extension__mbb0000__minimal_replicate_database.txz'
   #]

   bbdb = worms.database.BBlockDB(
      # cachedirs='test_extension_cache',
      lazy=False,
      **kw.sub(read_new_pdbs=True),
   )

   ssdag, result = worms.app.run_simple(criteria, **kw)

   # import pickle
   # with open(
   #       '/home/sheffler/src/worms/worms/data/test_cases/test_extension/testcache/test_extension_reference_results.pickle',
   #       'rb') as inp:
   #    crit, ssdag, result = pickle.load(inp)

   # for i in [None, 2, 4, 6, 8, 10, 12, 14]:
   for i in [None]:
      # for i in [None, 10]:
      iext = 0 if i is None else i
      extensions = dict() if i is None else {1: i}
      # extensions = dict()
      worms.app.output_simple(
         criteria,
         ssdag,
         result,
         sym='oct',
         extensions=extensions,
         # output_indices=[0],
         # output_prefix='/home/sheffler/src/worms/extension_cube',
         output_prefix='/home/sheffler/src/worms/test_ext/foo',
         output_suffix=str(iext),
         bblockdb=bbdb,
      )

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
