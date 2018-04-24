import os
import _pickle as pickle
from . import util
try:
    from pyrosetta import pose_from_file
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
except ImportError:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('pyrosetta not available, worms won\'t work')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


class Database:

    def __init__(self, cachefile='.worms_database.pickle'):
        self.cachefile = cachefile
        if os.path.exists(cachefile):
            with open(self.cachefile, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = dict(poses=dict(), coords=dict(), secstruct=dict)

        def _save(self):
            with open(self.cachefile, 'wb') as f:
                pickle.dump(self.cache, f)

        def get_pose(self, pdb_file):
            if pdb_file not in self.cache['poses']:
                pose = pose_from_file(pdb_file)
                self.cache['poses'][pdb_file] = pose
            return self.cache['poses'][pdb_file]

        def clear_poses(self):
            self.cache['poses'] = dict()

        def get_coords(self, pdb_file):
            if pdb_file not in self.cache['coords']:
                pose = self.get_pose(pdb_file)
                coords = util.get_bb_stubs(pose)
                self.cache['coords'][pdb_file] = pose
            return self.cache['coords'][pdb_file]

        def get_secstruct(self, pdb_file):
            pose = self.get_pose(pdb_file)
            ss = Dssp(pose).get_dssp_secstruct()
            self.cache['secstruct'][pdb_file] = ss

        def add_pdb_file(self, pdb_file):
            get_pose(pdb_file)
            get_coords(pdb_file)
            get_secstruct(pdb_file)

        def __del__(self):
            print('Database.__del__', id(self))
            self._save()
