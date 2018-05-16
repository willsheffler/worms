from worms.util import jit
from worms import KHashi8i8


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
