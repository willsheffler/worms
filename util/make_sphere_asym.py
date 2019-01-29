import _pickle

import numpy as np
from homog import sym
from worms.data import sphere
from worms.vis import plot3d
from worms.data.sphere import get_sphere_samples


def prune_to_asu(pts, frames):
    r = frames[..., :3, :3]
    x = r[:, None] @ pts[None, ..., None]
    d2 = np.sum((x.squeeze() - (1, 2, 3)) ** 2, axis=2)
    # print(frames.shape, pts.shape, x.shape, d2.shape)
    # print(np.min(d2), np.max(d2))
    closest = np.argmin(d2, axis=0)
    # print(np.sum(closest == 0), len(pts) / np.sum(closest == 0))

    for i in range(len(frames)):
        print("prune_to_asu", i, np.sum(closest == i), len(pts) / np.sum(closest == i))
    asu = pts[closest == 0]
    plot3d(asu)

    # test = r[:, None] @ pts[None, ..., None]
    # plot3d(test.reshape(-1, 3))

    return asu


def main():
    s = sphere.get_sphere_samples()

    # s_asym_tet = prune_to_asu(s, sym.tetrahedral_frames)
    # with open('sphere_tet_asu.pickle', 'wb') as out:
    #     _pickle.dump(s_asym_tet, out)

    # s_asym_oct = prune_to_asu(s, sym.octahedral_frames)
    # with open('sphere_oct_asu.pickle', 'wb') as out:
    #     _pickle.dump(s_asym_oct, out)

    # s_asym_ics = prune_to_asu(s, sym.icosahedral_frames)
    # with open('sphere_ics_asu.pickle', 'wb') as out:
    #     _pickle.dump(s_asym_ics, out)

    # return

    t = get_sphere_samples(sym="T")
    o = get_sphere_samples(sym="O")
    i = get_sphere_samples(sym="I")
    print("t", len(t), "o", len(o), "i", len(i))
    print("t", len(t) * 12, "o", len(o) * 24, "i", len(i) * 60)
    plot3d(t, norm=0)
    plot3d(o, norm=0)
    plot3d(i, norm=0)
    print(np.linalg.norm(t, axis=1).min())
    print(np.linalg.norm(t, axis=1).max())
    print(np.linalg.norm(o, axis=1).min())
    print(np.linalg.norm(o, axis=1).max())
    print(np.linalg.norm(i, axis=1).min())
    print(np.linalg.norm(i, axis=1).max())


if __name__ == "__main__":
    main()
