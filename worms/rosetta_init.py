import sys
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose import append_pose_to_pose, append_subpose_to_pose

core.pose.Pose.__len__ = core.pose.Pose.size

_initargs = ' '.join([
   "-mute all",
   "-corrections:beta_nov16",
   "-beta",
   "-preserve_crystinfo",
   "-symmetry::symmetry_definition dummy",
   # "--prevent_repacking",
])

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

def pose_from_str(s):
   p = Pose()
   core.import_pose.pose_from_pdbstring(p, s)
   return p
