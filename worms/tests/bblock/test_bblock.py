import numpy as np
from worms.bblock import ncac_to_stubs

def test_bblock(bbdb):
   for bb in bbdb.query("all"):
      assert bblock_str(bb).startswith("jitclass")

def test_ncac_to_stubs():
   ncac = np.random.randn(10, 3, 4).reshape(10, 3, 4)
   stubs = ncac_to_stubs(ncac)
   assert hm.is_homog_xform(stubs)

if __name__ == '__main__':
   test_make_connections_merge()
