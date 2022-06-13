import numpy as np
import worms
from worms.bblock.bbutil import ncac_to_stubs, bblock_str
import worms.homog as hm

def test_bblock(db_bblock_caching_v0):
   for bb in db_bblock_caching_v0.query("all"):
      assert bblock_str(bb).startswith("jitclass")

def test_ncac_to_stubs():
   ncac = np.random.randn(10, 3, 4).reshape(10, 3, 4)
   stubs = ncac_to_stubs(ncac)
   # print(stubs)
   assert hm.is_homog_xform(stubs)

if __name__ == '__main__':
   db_bblock_caching_v0 = worms.data.db_bblock_caching_v0()
   test_bblock(db_bblock_caching_v0)
   test_ncac_to_stubs()
