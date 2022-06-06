import os

if __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'
   print('NUMBA_DISABLE_JIT')

import worms

def test_p213():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   worms.tests.generic_integration_test('test_p213')

if __name__ == '__main__':
   test_p213()
