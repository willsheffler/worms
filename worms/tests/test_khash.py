from worms.util import jit
from worms import KHashi8i8
from worms.khash.khash_cffi import _khash_get


@jit
def use_khash_jitclass(h, i):
    return h.get(i) + 10


def test_khash_jitclass():
    h = KHashi8i8()
    h.update([(7, 3), (13, 4)])
    h.set(1, 13)
    assert h.get(1) == 13
    assert h.get(7) == 3
    assert h.get(13) == 4
    assert use_khash_jitclass(h, 1) == 23
    assert use_khash_jitclass(h, 7) == 13
    assert use_khash_jitclass(h, 13) == 14
    assert h.get(-2345) == -9223372036854775808
    assert h.get(926347) == -9223372036854775808
    assert h.size() == 3


@jit
def numba_get(h, i):
    return h.get(i)


def foo(h):
    hash = h.hash

    @jit
    def func(i):
        return _khash_get(hash, i, -9223372036854775808)

    return func


def test_khash_numba_closure():
    h = KHashi8i8()
    h.set(10, 10)

    assert numba_get(h, 10) == 10

    f = foo(h)
    assert f(10) == 10
