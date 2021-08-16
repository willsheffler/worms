# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np

from worms import homog as hg

from glumpy.api.matplotlib import *
from glumpy.transforms import *

# Create a new figure
figure = Figure((24, 12))

tball = Trackball(name="trackball")

# Create a subplot on left, using trackball interface (3d)
left = figure.add_axes([0.0, 0.0, 0.5, 1.0], interface=tball, facecolor=(1, 1, 1, 0.25))

# Create a subplot on right, using panzoom interface (2d)
right = figure.add_axes(
   [0.5, 0.0, 0.5, 1.0],
   xscale=LinearScale(),
   yscale=LinearScale(),
   zscale=LinearScale(),
   interface=tball,
   facecolor=(1, 1, 1, 0.25),
)

# Create a new collection of points
collection1 = PointCollection("agg")
collection2 = PointCollection("agg")

left.add_drawable(collection1)
right.add_drawable(collection2)

dat1 = np.random.uniform(-2, 2, (10000, 3))
dat2 = (hg.rot([0, 0, 1], -0.08) @ dat1[..., None]).squeeze()

collection1.append(dat1)
collection2.append(dat2)

# Show figure
figure.show()
