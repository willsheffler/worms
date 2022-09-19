import os

if False and __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'
   print('NUMBA_DISABLE_JIT')

import worms
import willutil as wu

def test_p6m_D2_C3():
   name = 'test_p6m_D2_C3'
   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   # worms.tests.generic_integration_test('test_p6m')

   argv = ['@' + worms.data.test_file_path(f'{name}/config/{name}.flags')]
   # argv = ['@' + worms.data.test_file_path(f'test_extension/config/test_extension.flags')]
   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)
   kw.timer = wu.Timer()
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   savefn = f'{name}_result.pickle'

   # ssdag, result = wu.load(savefn)
   ssdag, result = worms.app.run_simple(criteria, **kw)
   wu.save((ssdag, result), savefn)

   result = worms.filters.check_geometry(ssdag, criteria, result, **kw)

   worms.output.filter_and_output_results(
      criteria,
      ssdag,
      result,
      # use_simple_pose_construction=False,
      # **kw.sub(output_from_pose=True, merge_bblock=0, output_prefix='testout_orig/testout_orig',
      # ignore_recoverable_errors=False),
      use_simple_pose_construction=False,
      # use_simple_pose_construction=True,
      **kw.sub(
         output_from_pose=True,
         merge_bblock=0,
         # output_prefix=f'{name}/{name}_',
         ignore_recoverable_errors=False,
         # only_outputs=[0, 1, 2, 3],
      ),
   )

if __name__ == '__main__':
   test_p6m_D2_C3()
