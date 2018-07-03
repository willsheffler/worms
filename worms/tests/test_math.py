from worms.math import *
import homog as hm
import numpy as pn
import time
import pytest


def test_numba_axis_angle_of():
    x = hm.rand_xform((100, ))

    tn = time.time()
    axs, ang = numba_axis_angle_of(x)
    tn = time.time() - tn
    tp = time.time()
    refaxs, refang = hm.axis_angle_of(x)
    tp = time.time() - tp
    print(tn, tp)

    assert np.allclose(refaxs, axs)
    assert np.allclose(refang, ang)


def test_numba_point_in_plane():
    plane = hm.rand_ray((100, ))
    for i in range(100):
        assert numba_point_in_plane(plane[i], plane[i, :3, 0])
    pt = hm.proj_perp(plane[..., :3, 1], np.random.randn(3))
    for i in range(100):
        assert numba_point_in_plane(plane[i], plane[i, :3, 0] + pt[i])


@pytest.mark.skip
def test_intersect_planes():
    with pytest.raises(ValueError):
        numba_intersect_planes(
            np.array([[0, 0, 0, 2], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T
        )
    with pytest.raises(ValueError):
        numba_intersect_planes(
            np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T,
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T
        )
    # with pytest.raises(ValueError):
    # numba_intersect_planes(
    # np.array([[0, 0, 1], [0, 0, 0, 0]]).T,
    # np.array([[0, 0, 1], [0, 0, 0, 1]]).T
    # )

    # isct, sts = numba_intersect_planes(np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]),
    # np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]))
    # assert isct.shape[:-2] == sts.shape == (9,)
    # assert np.all(sts == 2)

    # isct, sts = numba_intersect_planes(np.array([[1, 0, 0, 1], [1, 0, 0, 0]]),
    # np.array([[0, 0, 0, 1], [1, 0, 0, 0]]))
    # assert sts == 1

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T
    )
    assert isct[2, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 0, 1))

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T
    )
    # assert sts == 0
    assert isct[1, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (0, 1, 0))

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T,
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T
    )
    assert isct[0, 0] == 0
    assert np.all(abs(isct[:3, 1]) == (1, 0, 0))

    isct = numba_intersect_planes(
        np.array([[7, 0, 0, 1], [1, 0, 0, 0]]).T,
        np.array([[0, 9, 0, 1], [0, 1, 0, 0]]).T
    )
    assert np.allclose(isct[:3, 0], [7, 9, 0])
    assert np.allclose(abs(isct[:3, 1]), [0, 0, 1])

    isct = numba_intersect_planes(
        np.array([[0, 0, 0, 1], hm.hnormalized([1, 1, 0, 0])]).T,
        np.array([[0, 0, 0, 1], hm.hnormalized([0, 1, 1, 0])]).T
    )
    assert np.allclose(abs(isct[:, 1]), hm.hnormalized([1, 1, 1]))

    p1 = hm.hray([2, 0, 0, 1], [1, 0, 0, 0])
    p2 = hm.hray([0, 0, 0, 1], [0, 0, 1, 0])
    isct = numba_intersect_planes(p1, p2)
    assert np.all(hm.ray_in_plane(p1, isct))
    assert np.all(hm.ray_in_plane(p2, isct))

    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
                   [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
                   [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
    isct = numba_intersect_planes(p1, p2)
    assert np.all(hm.ray_in_plane(p1, isct))
    assert np.all(hm.ray_in_plane(p2, isct))


def test_numba_intersect_planes_rand():
    # origin case
    plane1, plane2 = hm.rand_ray(shape=(2, 1))
    plane1[..., :3, 0] = 0
    plane2[..., :3, 0] = 0
    isect = numba_intersect_planes(plane1.squeeze(), plane2.squeeze())
    assert np.all(hm.ray_in_plane(plane1, isect))
    assert np.all(hm.ray_in_plane(plane2, isect))

    # orthogonal case
    plane1, plane2 = hm.rand_ray(shape=(2, 1))
    plane1[..., :, 1] = hm.hnormalized([0, 0, 1])
    plane2[..., :, 1] = hm.hnormalized([0, 1, 0])
    isect = numba_intersect_planes(plane1.squeeze(), plane2.squeeze())
    assert np.all(hm.ray_in_plane(plane1, isect))
    assert np.all(hm.ray_in_plane(plane2, isect))

    # general case
    for i in range(len(plane1)):
        plane1 = hm.rand_ray(shape=(1, )).squeeze()
        plane2 = hm.rand_ray(shape=(1, )).squeeze()
        isect = numba_intersect_planes(plane1, plane2)
        assert np.all(hm.ray_in_plane(plane1, isect))
        assert np.all(hm.ray_in_plane(plane2, isect))
