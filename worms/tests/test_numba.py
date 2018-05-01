import numpy as np
import pytest
import numba as nb
import random
from timeit import timeit
import toolz
from collections import namedtuple
from worms.khash import *
from worms.khash.khash_cffi import _khash_init, _khash_set, _khash_get, _khash_destroy

T = namedtuple('T', 'a b c d'.split())


@nb.njit
def numba_named_tuple(t):
    s = T(0, 1, 2, 3)
    return (t.a * s.a +
            t.b * s.b +
            t.c * s.c +
            t.d * s.d )  # yapf: disable

def test_numba_named_tuple():
    t = T(1, 2, 3, 4)
    assert numba_named_tuple(t) is 20


@nb.jit
def without_khash(fid, values, fetch_ids):
    # Build map of fid's (non-continuous) to fix (continuous compact)
    fid2fix = np.zeros(np.max(fid) + 1, dtype=np.int64)
    fid2fix[np.unique(fid)] = np.arange(len(np.unique(fid)), dtype=np.int64)

    # Now fetch a selection of values
    s = np.empty_like(fetch_ids, dtype=np.float64)
    for i in range(fetch_ids.shape[0]):
        ii = fid2fix[fetch_ids[i]]
        s[i] = values[ii]

    return s


def with_khash(fid, values, fetch_ids):
    d = _khash_init()

    fix = 0
    for i in range(fid.shape[0]):
        _khash_set(d, fid[i], fix)
        fix += 1

    s = np.empty_like(fetch_ids, dtype=np.float64)
    for j in range(fetch_ids.shape[0]):
        ii = _khash_get(d, fetch_ids[j], -99)
        s[j] = values[ii]

    _khash_destroy(d)
    return s


with_khash_numba = nb.njit()(with_khash)


def disabled_test_khash():
    max_fid = 215000
    n_fids = 130
    n_fetch = 100

    _fids = np.arange(max_fid)
    np.random.shuffle(_fids)
    fids = np.empty(n_fids, dtype=np.int64)
    fids[-1] = max_fid
    fids[:-1] = np.sort(_fids[:n_fids - 1])

    values = np.random.normal(size=(n_fids))
    fetch_ids = np.random.choice(fids, size=(n_fetch, ), replace=True)

    s1 = without_khash(fids, values, fetch_ids)
    s2 = with_khash_numba(fids, values, fetch_ids)
    s3 = with_khash(fids, values, fetch_ids)
    assert np.allclose(s1, s2)
    assert np.allclose(s1, s3)


@nb.njit
def numba_take_hash(d):
    return _khash_get(d, 1) + _khash_get(d, 2)


@nb.njit
def numba_make_hash():
    d = _khash_init()
    _khash_set(d, 1, 1)
    _khash_set(d, 2, 2)
    return d


@pytest.mark.xfail
def test_khash_data_passing_in():

    from worms.khash import _ffi
    d = _khash_init()
    print('py', d)
    print('cffi', _ffi.typeof(d))
    print(repr(_ffi.typeof(d)))
    print(type(_ffi.typeof(d)))

    # assert 0

    d = _khash_init()
    s = numba_take_hash(d)
    _khash_destroy(d)


@pytest.mark.xfail
def test_khash_data_passing_out():
    d = _khash_init()
    print('python khash', nb.typeof(d))

    d = numba_make_hash()
    print('returned from numba', nb.typeof(d))
    # v = numba_take_hash(d)
    assert v == 3


@nb.njit
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
