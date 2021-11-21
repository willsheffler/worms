import sys, os
from worms.app.main import worms_main
from worms.database.merge import merge_main

if __name__ == "__main__":

   print("RUNNING WORMS CWD", os.getcwd())
   assert len(sys.argv) > 1
   if sys.argv[1] == 'collect':
      mergefname = sys.argv[2]
      assert not mergefname.endswith('.txz')
      merge_main(mergefname + '.txz', sys.argv[3:])
   else:
      worms_main(sys.argv[1:])
