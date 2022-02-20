import os

if __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'

import worms

def test_i213():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   # criteria, kw = worms.tests.setup_testfunc('test_i213')
   # kw.verbosity = 0
   # ssdag, newresult = worms.app.run_simple(criteria, **kw)
   # worms.app.output_simple(criteria, ssdag, newresult, **kw)
   # assert 0

   worms.tests.generic_integration_test('test_i213')

if __name__ == '__main__':
   test_i213()
