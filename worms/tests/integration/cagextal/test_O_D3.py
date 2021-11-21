import os

if __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'

import worms

def test_cagextal_O_D3():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   worms.tests.generic_integration_test('test_cagextal_O_D3')

if __name__ == '__main__':
   test_cagextal_O_D3()
