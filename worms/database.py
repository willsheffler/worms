import os
import json
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

    def __init__(self, json_db_files, *,
                 cachefile='.worms_database.pickle', cache_only=False):
        self.cachefile = cachefile
        self.json_db_files = json_db_files
        if os.path.exists(cachefile):
            with open(self.cachefile, 'rb') as f:
                self.cache = pickle.load(f)
        elif cache_only:
            raise ValueError('no cachefile found', cachefile)
        else:
            self.cache = dict(poses=dict(), coords=dict(), secstruct=dict())

        for json_db_file in json_db_files:
            with open(json_db_file) as f:
                db = json.load(f)
                # print('-' * 100)
                print(json_db_file)
                for entry in db:
                    self.add_pdb_file(entry['file'])
                    return
                    # print(entry['file'])
                    # assert 0

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
        self.get_pose(pdb_file)
        self.get_coords(pdb_file)
        self.get_secstruct(pdb_file)

    def save(self):
        with open(self.cachefile, 'wb') as f:
            pickle.dump(self.cache, f)


class DatabaseContext:

    def __init__(self, json_db_files, **kw):
        self.json_db_files = json_db_files
        self.kw = kw

    def __enter__(self):
        self.db = Database(self.json_db_files, **self.kw)
        return self.db

    def __exit__(self, *args):
        self.db.save()
