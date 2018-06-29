from worms.math import *
import homog as hm
import numpy as pn
import time


def test_math():
    x = hm.rand_xform((30, 40, 50, 60))
    axs, ang = numba_axis_angle_of(x[1, 1, 1])

    tn = time.time()
    axs, ang = numba_axis_angle_of(x)
    tn = time.time() - tn
    tp = time.time()
    refaxs, refang = hm.axis_angle_of(x)
    tp = time.time() - tp
    print(tn, tp)

    assert np.allclose(refaxs, axs)
    assert np.allclose(refang, ang)
