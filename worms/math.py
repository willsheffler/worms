from worms.util import jit
import numpy as np


@jit
def clip(val, mn, mx):
    return np.maximum(np.minimum(val, mx), mn)


@jit
def trace44(xforms):
    return xforms[0, 0] + xforms[1, 1] + xforms[2, 2] + xforms[3, 3]


@jit
def numba_axis_angle_single(xform):
    axs = np.zeros((4, ))
    axs[0] = xform[2, 1] - xform[1, 2]
    axs[1] = xform[0, 2] - xform[2, 0]
    axs[2] = xform[1, 0] - xform[0, 1]
    four_sin2 = np.sum(axs**2, axis=-1)
    norm = np.sqrt(four_sin2)
    axs = axs / norm
    sin_angl = clip(norm / 2.0, -1, 1)
    trace = trace44(xform) / 2 - 1
    cos_angl = clip(trace, -1, 1)
    ang = np.arctan2(sin_angl, cos_angl)
    return axs, ang


@jit
def numba_axis_angle_of(xforms):
    shape = xforms.shape[:-2]
    xforms = xforms.reshape(-1, 4, 4)
    n = len(xforms)
    axs = np.empty((n, 4), dtype=xforms.dtype)
    ang = np.empty((n, ), dtype=xforms.dtype)

    for i in range(n):
        axs[i], ang[i] = numba_axis_angle_single(xforms[i])

    axs = axs.reshape(*shape, 4)
    ang = ang.reshape(*shape)
    return axs, ang
