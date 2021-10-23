from deferred_import import deferred_import

pyrosetta = deferred_import('pyrosetta')
ros = deferred_import('pyrosetta.rosetta')
util = deferred_import('worms.util.rosetta_utils')

def pack_filter(pose, splices):
   print(len(pose.residues), splices)
