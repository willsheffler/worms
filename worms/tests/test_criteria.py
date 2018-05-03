from worms.criteria import *
from homog import *


def test_geom_check():
    SX = Cyclic
    I = np.identity(4)
    rotx1rad = hrot([1, 0, 0], 1)
    transx10 = htrans([10, 0, 0])
    randaxes = np.random.randn(1, 3)

    assert 0 == SX('c1').score([I, I])
    assert 0.001 > abs(50 - SX('c1').score([I, rotx1rad]))
    assert 1e-5 > abs(SX('c2').score([I, hrot([1, 0, 0], np.pi)]))

    score = Cyclic('c2').score([I, hrot(randaxes, np.pi)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)

    score = Cyclic('c3').score([I, hrot(randaxes, np.pi * 2 / 3)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)

    score = Cyclic('c4').score([I, hrot(randaxes, np.pi / 2)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)
