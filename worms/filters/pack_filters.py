from pyrosetta import rosetta as ros


def pack_filter(pose, splices):
    print(len(pose.residues), splices)
