import numpy as np

import homog as hg

from glumpy.api.matplotlib import *
from glumpy.transforms import *


def plot3d(data):

    com = data.mean(axis=0)
    data -= com
    rg = np.sqrt(np.sum(data**2) / len(data))
    print(com)
    print(rg)
    data /= rg

    figure = Figure((24, 12))
    # use shared Trackball iface
    tball = Trackball(name="trackball")
    left = figure.add_axes(
        [0.0, 0.0, 0.5, 1.0],
        interface=tball,
        facecolor=(1, 1, 1, 0.25),
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
