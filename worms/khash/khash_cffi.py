import numba as nb
import numpy as np
from numba import cffi_support
import glob
import os
from cffi import FFI
from numba import cffi_support

# from cffi import _cffi_backend

_ffi = FFI()
_ffi.set_source("_khash_ffi", '#include "khash_int2int.h"')

_ffi.cdef(
    """\
typedef int... khint64_t;

static inline void *khash_int2int_init(void);
static void khash_int2int_destroy(void *);
static inline khint64_t khash_int2int_get(void *, khint64_t, khint64_t);
static inline int khash_int2int_set(void *, khint64_t, khint64_t);
static inline int khash_int2int_size(void *);
"""
)

_dir = os.path.dirname(__file__)
try:
    from worms.khash import _khash_ffi
except ImportError:
    print("worms.khash first run, building...")
    _ffi.compile(tmpdir=_dir)

if not "READTHEDOCS" in os.environ:
    from worms.khash import _khash_ffi

    _khash_init = _khash_ffi.lib.khash_int2int_init
    _khash_get = _khash_ffi.lib.khash_int2int_get
    _khash_set = _khash_ffi.lib.khash_int2int_set
    _khash_size = _khash_ffi.lib.khash_int2int_size
    _khash_destroy = _khash_ffi.lib.khash_int2int_destroy

    # cffi_support.register_type(
    # _ffi.typeof(_khash_init()),
    # nb.types.voidptr,
    # )
    cffi_support.register_module(_khash_ffi)


@nb.jitclass((("hash", nb.types.voidptr),))
class KHashi8i8:
    def __init__(self):
        self.hash = _khash_init()

    def update(self, iterable):
        for i in iterable:
            self.set(*i)

    def update2(self, ary, ary2):
        assert len(ary) == len(ary2)
        for i in range(len(ary)):
            self.set(ary[i], ary2[i])

    def get(self, i):
        return _khash_get(self.hash, i, -9223372036854775808)

    def set(self, i, v):
        return _khash_set(self.hash, i, v)

    def size(self):
        return _khash_size(self.hash)

    def __del__(self):
        _khash_destroy(self.hash)
