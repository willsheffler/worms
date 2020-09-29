import numpy as np
import os
import glob
import _pickle
from worms.util import hash_str_to_int

def _convert_from_pdb():
   """almost the same, dicts little slower, little bigger"""
   for f in glob.glob("/home/sheffler/.worms/cache/splices.bk/*/*.pickle"):
      pdbfile = os.path.basename(f).replace("__", "/")[:-7]
      pdbkey = hash_str_to_int(pdbfile)
      newf = ("/home/sheffler/.worms/cache/splices/" + "%016x" % 5633173723268761018 + "/" +
              "%016x" % pdbkey + ".pickle")
      newcachhe = dict()
      with open(f, "rb") as inp:
         cache = _pickle.load(inp)
         for k0, v0 in cache.items():
            assert len(v0) == 2
            assert isinstance(v0[0], np.ndarray)
            assert isinstance(v0[1], np.ndarray)
            newcachhe[hash_str_to_int(k0)] = v0
      with open(newf, "wb") as out:
         _pickle.dump(newcachhe, out)

if __name__ == "__main__":
   _convert_from_pdb()
