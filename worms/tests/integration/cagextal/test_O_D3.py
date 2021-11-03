import sys, os, pickle, inspect

# from worms.util.util import _disable_jit

# _disable_jit()
os.environ['NUMBA_DISABLE_JIT'] = '1'

import worms

# worms.util.util.jit = worms.util.disabled_jit
# worms.util.util.jitclass = worms.util.disabled_jitclass
# worms.util.util.priority_jit = worms.util.disabled_priority_jit

def test_cagextal_O_D3():

   # assumes
   # test dir worms/data/test_cases/<testname>/
   # reference results in that dir
   # flags: worms/data/test_cases/<testname>/config/<testname>.flags

   should pdbs get checked in??? can reliably regenerate them?

   worms.tests.generic_integration_test('test_cagextal_O_D3')

if __name__ == '__main__':
   test_cagextal_O_D3()
# tmp_check_pdbout()
