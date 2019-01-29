import numpy as np

import homog as hg

try:
    from glumpy.api.matplotlib import *
    from glumpy.transforms import *

    _have_glumpy = True
except ImportError:
    _have_glumpy = False

try:
    import matplotlib.pyplot as plt

    _have_matplotlib = True
except ImportError:
    _have_matplotlib = False


def scatter(*args, **kwargs):
    plt.scatter(*args, **kwargs)
    plt.show()


def plot3d(data, norm=True):
    if not _have_glumpy:
        print("plot3d: glumpy not available, no 3d plotting")
        return

    if norm:
        data = data.copy()
        com = data.mean(axis=0)
        data -= com
        rg = np.sqrt(np.sum(data ** 2) / len(data))
        print("plot3d com", com)
        print("plot3d rg", rg)
        data /= rg

    figure = Figure((24, 12))
    # use shared Trackball iface
    tball = Trackball(name="trackball")
    left = figure.add_axes(
        [0.0, 0.0, 0.5, 1.0], interface=tball, facecolor=(1, 1, 1, 0.25)
    )
    right = figure.add_axes(
        [0.5, 0.0, 0.5, 1.0],
        xscale=LinearScale(),
        yscale=LinearScale(),
        zscale=LinearScale(),
        interface=tball,
        facecolor=(1, 1, 1, 0.25),
    )
    collection1 = PointCollection("agg")
    collection2 = PointCollection("agg")
    left.add_drawable(collection1)
    right.add_drawable(collection2)

    dat2 = (hg.rot([0, 0, 1], -0.08) @ data[..., None]).squeeze()

    collection1.append(data)
    collection2.append(dat2)

    # Show figure
    figure.show()
