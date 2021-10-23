import sys
from pyrosetta import rosetta, init

_initargs = "-corrections:beta_nov16 -mute all -preserve_crystinfo -symmetry::symmetry_definition dummy -beta --prevent_repacking"
print('============================ ROSETTA INIT ============================')
init(_initargs)
print('======================================================================')
sys.stdout.flush()
sys.stderr.flush()
# _initargs = None

def rosetta_init_safe(args):
   global _initargs
   if _initargs is None:
      init(args)
      _initargs = args
   if _initargs != args:
      raise ValueError('rosetta init args mismatch')

def is_rosetta_initialized():
   return _initargs is None
