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


@jit
def numba_is_valid_ray(r):
    if r.shape != (4, 2): return False
    if (r[3, 0] != 1 or r[3, 1] != 0): return False
    # if abs(np.linalg.norm(r[..., :3, 1], axis=-1) - 1) > 0.000001:
    # return False
    return True


@jit
def numba_dot(u, v):
    return np.sum(u * v)


@jit
def numba_point_in_plane(plane, pt):
    return np.abs(numba_dot(plane[:3, 1], pt[:3] - plane[:3, 0])) < 0.000001


@jit
def numba_cross(u, v):
    result = np.empty((3, ), dtype=u.dtype)
    result[0] = u[1] * v[2] - u[2] * v[1]
    result[1] = u[2] * v[0] - u[0] * v[2]
    result[2] = u[0] * v[1] - u[1] * v[0]
    return result


@jit
def numba_intersect_planes(plane1, plane2):
    """intersect_Planes: find the 3D intersection of two planes
       Input:  two planes represented by rays shape=(..., 4, 2)
       Output: *L = the intersection line (when it exists)
       Return: rays shape=(...,4,2), status
               0 = intersection returned
               1 = disjoint (no intersection)
               2 = the two planes coincide
    """
    if not numba_is_valid_ray(plane1): raise ValueError('invalid plane1')
    if not numba_is_valid_ray(plane2): raise ValueError('invalid plane2')
    p1, n1 = plane1[:3, 0], plane1[:3, 1]
    p2, n2 = plane2[:3, 0], plane2[:3, 1]
    u = numba_cross(n1, n2)
    abs_u = np.abs(u)
    planes_parallel = np.sum(abs_u, axis=-1) < 0.000001
    p2_in_plane1 = numba_point_in_plane(plane1, p2)
    if planes_parallel:
        return None
    # if p2_in_plane1:
    # return None
    d1 = -np.sum(n1 * p1)
    d2 = -np.sum(n2 * p2)
    amax = np.argmax(abs_u)
    if amax == 0:
        x = 0
        y = (d2 * n1[2] - d1 * n2[2]) / u[0]
        z = (d1 * n2[1] - d2 * n1[1]) / u[0]
    elif amax == 1:
        x = (d1 * n2[2] - d2 * n1[2]) / u[1]
        y = 0
        z = (d2 * n1[0] - d1 * n2[0]) / u[1]

    elif amax == 2:
        x = (d2 * n1[1] - d1 * n2[1]) / u[2]
        y = (d1 * n2[0] - d2 * n1[0]) / u[2]
        z = 0
    isect = np.empty((4, 2), dtype=plane1.dtype)
    isect[0, 0] = x
    isect[1, 0] = y
    isect[2, 0] = z
    isect[3, 0] = 1
    isect[:3, 1] = u / np.sqrt(np.sum(u * u))
    isect[3, 1] = 0
    return isect


@jit
def numba_axis_angle_cen_single(xform):
    axis, angle = numba_axis_angle_single(xform)
    #  sketchy magic points...
    p1 = (-32.09501046777237, 03.36227004372687, 35.34672781477340, 1)
    p2 = (21.15113978202345, 12.55664537217840, -37.48294301885574, 1)
    # p1 = rand_point()
    # p2 = rand_point()
    tparallel = hdot(axis, xforms[..., :, 3])[..., None] * axis
    q1 = xform @ p1 - tparallel
    q2 = xform @ p2 - tparallel
    n1 = hnormalized(q1 - p1)
    n2 = hnormalized(q2 - p2)
    c1 = (p1 + q1) / 2.0
    c2 = (p2 + q2) / 2.0
    plane1 = hray(c1, n1)
    plane2 = hray(c2, n2)
    isect, status = intersect_planes(plane1, plane2)
    return axis, angle, isect[..., :, 0]
