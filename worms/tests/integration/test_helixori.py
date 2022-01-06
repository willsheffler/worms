import os

if __name__ == '__main__':
   os.environ['NUMBA_DISABLE_JIT'] = '1'

import worms

def test_helixori():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   worms.tests.generic_integration_test('test_helixori')

def test_helixori_nullcriteria():
   worms.tests.generic_integration_test('test_helixori_null')

if __name__ == '__main__':
   # test_helixori()
   test_helixori_nullcriteria()
