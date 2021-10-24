from deferred_import import deferred_import

ros = deferred_import('worms.rosetta_init')
util = deferred_import('worms.util.rosetta_utils')

def pack_filter(pose, splices):
   print(len(pose.residues), splices)
