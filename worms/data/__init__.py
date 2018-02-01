import os

try:
    import pyrosetta
except ImportError:
    print('no pyrosetta!')
from functools import lru_cache


class PoseLib:

    @lru_cache()
    def get(self, name):
        if name.startswith('__'): return
        this_dir, this_filename = os.path.split(__file__)
        return pyrosetta.pose_from_file(os.path.join(this_dir, name + '.pdb'))

    def __getattr__(self, name):
        return self.get(name)


poselib = PoseLib()
