import numpy as np

from pyrosetta import pose_from_file

from worms import util


class ClashGrid:
    def __init__(self, pdbfile, clashdis=3.5, spacing=0.5, **kw):
        self.clashdis = clashdis
        self.spacing = spacing
        bbdb = kw['db'][0]
        print('ClashGrid: reading pose from bbdb', pdbfile)
        # pose takes too much mem and too hard to serialize, dont store
        pose = bbdb.pose(pdbfile)
        assert pose is not None
        bbdb.savepose(pdbfile)
        # print('extracting coords')
        self.ncac = util.get_bb_coords(pose).reshape(-1, 4)[:, :3]
        # print(self.ncac.shape)
        mn = self.ncac.min(axis=0) - self.clashdis
        mx = self.ncac.max(axis=0) + self.clashdis
        shape = ((mx - mn) / self.spacing).astype('i4') + 1
        # print(mn, mx, shape)
        self.grid = np.zeros(shape, dtype='i1')
        self.lb = mn
        # print(self.grid.nbytes / 1000000)
        crdidx = ((self.ncac - self.lb) / self.spacing).astype('i4')
        # print(self.grid.shape, crdidx.min(0), crdidx.max(0))
        n = int(clashdis / spacing)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    if i**2 + j**2 + k**2 <= n**2:
                        idx = crdidx + (i, j, k)
                        self.grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        # print('volfrac', np.sum(self.grid) / self.grid.size)
        print('built ClashGrid', n, self.grid.shape, pdbfile)

    def clashcheck(self, xyz):
        if xyz.shape[-2:] == (3, 4):
            xyz = xyz.reshape(-1, 4)[:, :3]
        c = ((xyz - self.lb) / self.spacing).astype('i4')
        ok1 = np.all(c >= 0, axis=1)
        ok2 = np.all(c < self.grid.shape, axis=1)
        ok = ok1 * ok2
        c = c[ok]
        return np.sum(self.grid[c[:, 0], c[:, 1], c[:, 2]])