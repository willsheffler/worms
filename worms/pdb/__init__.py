import pyrosetta
from functools import lru_cache


class PoseLib:

    @lru_cache()
    def get(self, name):
        if name.startswith('__'):
            return
        rcl.init_check('-mute all', strict=False)
        this_dir, this_filename = os.path.split(__file__)
        pdb_dir = os.path.join(this_dir, 'pdb')
        return pyrosetta.pose_from_file(os.path.join(pdb_dir, name + '.pdb'))

    def __getattr__(self, name):
        return self.get(name)


poselib = PoseLib() if rcl.HAVE_PYROSETTA else None
