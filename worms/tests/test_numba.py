import numpy as np
import pytest
import numba as nb
import numba.types as nt
import random
from timeit import timeit
import toolz
from collections import namedtuple
from worms.khash import *
from worms.khash.khash_cffi import _khash_init, _khash_set, _khash_get, _khash_destroy
from worms.util import jit
from worms.tests import only_if_jit

T = namedtuple("T", "a b c d".split())

# @only_if_jit
# def test_numba_iterate():
#    @jit
#    def numba_sum(l):
#       tot = 0
#       for i in l:
#          tot += i
#       return tot

#    assert numba_sum(np.array([1, 2, 3])) == 6
#    assert numba_sum(np.array((1, 2, 3))) == 6
#    assert numba_sum(np.arange(1, 4)) == 6

#    @jit
#    def numba_sum2(k, l):
#       tot = 0
#       for i in k:
#          for j in l:
#             tot += i * j
#       return tot

#    for u in ([1, 2, 3], (1, 2, 3), np.arange(1, 4)):
#       for v in ([0, 1, 2], (0, 1, 2), np.arange(0, 3)):
#          assert numba_sum2(u, v) == 18
#    assert len(numba_sum2.nopython_signatures) == 9

@only_if_jit
def test_numba_reshape():
   @jit
   def numba_reshape(m44, a):
      tmp = a.reshape(-1, 4, 1)
      for i in range(len(tmp)):
         tmp[i] = m44 @ tmp[i]
      return tmp.reshape(-1, 3, 4)

   m44 = np.eye(4, dtype="f4")
   ncac = np.arange(120, dtype="f4").reshape(-1, 3, 4)
   ncac[..., 3] = 1
   assert numba_reshape(m44, ncac).shape == (10, 3, 4)

@jit
def numba_named_tuple(t):
   s = T(0, 1, 2, 3)
   return t.a * s.a + t.b * s.b + t.c * s.c + t.d * s.d  # yapf: disable


@only_if_jit
def test_numba_named_tuple():
   t = T(1, 2, 3, 4)
   assert numba_named_tuple(t) is 20

@jit
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

with_khash_numba = nb.njit(nogil=1)(with_khash)

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

@pytest.mark.skip
def test_many_specs():
   @jit
   def add(u):
      return u[0] + u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7]

   import itertools as it

   n = 8
   binary = np.right_shift(np.arange(2**n)[:, None], np.arange(n)[None]) % 2
   tups = [tuple([(1, 1.0)[i] for i in x]) for x in binary]
   for x in tups:
      add(x)

   print("num sigs", len(add.nopython_signatures))

   assert 0

# @only_if_jit
# def test_jitclass_print():
#     @nb.experimental.jitclass((('add', nt.int32), ))
#     class DummyFilter:
#         def __init__(self, add):
#             self.add = add
#
#         def jitclass_members_still_cant_print(self):
#             print('literal')  # this is ok
#             print(self.add)  # these are not
#             print(np.eye(4))
#             return 13
#
#     # cannot subclass from a jitclass
#     # @nb.experimental.jitclass((('add', nt.int32), ))
#     # class DummyFilter2(DummyFilter):
#     # pass
#     df = DummyFilter(13)
#     with pytest.raises(RuntimeError):
#         assert df.jitclass_members_still_cant_print() == 13

@only_if_jit
def test_numba_tuple_of_arrays():
   @jit
   def expand_tuples_of_arrays(tup):
      new = tup + (tup[0][:], )
      return new
      # return expand_tuples_of_arrays(new) # crashes...

   tup = tuple(np.arange(7) for i in range(4))
   assert len(tup) is 4
   tup2 = expand_tuples_of_arrays(tup)
   assert len(tup2) is 5
   tup3 = expand_tuples_of_arrays(tup2)
   assert len(tup3) is 6

# functions are not first class in numba yet
# @nb.experimental.jitclass((('dummy', nt.int32), ))
# class AlwaysOne:
#     def __init__(self):
#         self.dummy = 1
#
#     def call(self, arg):
#         return np.ones(len(arg))
#
#
# @nb.experimental.jitclass((('dummy', nt.int32), ))
# class AlwaysZero:
#     def __init__(self):
#         self.dummy = 1
#
#     def call(self, arg):
#         return np.zeros(len(arg))
#
#
# @jit
# def run_funcs(callers, arg):
#     print(len(callers))
#
#     result = np.empty(shape=(len(callers), len(arg)), dtype=np.float64)
#     for func in callers:
#         r = func.call(arg)
#         # print(func)
#         # result[i] = r
#     return result
#
#
# def test_tuple_of_jitfuncs():
#     callers = (AlwaysZero(), AlwaysOne())
#     print(callers[0].call(np.arange(4)))
#     print(nb.typeof(callers[1]))
#     result = run_funcs(callers, np.arange(10))
#     print(result)
#     assert 0

@only_if_jit
def test_numba_chain_funcs():
   @nb.njit
   def ident(x):
      return x

   def chain(fs, inner=ident):
      head, tail = fs[-1], fs[:-1]

      @nb.njit
      def wrap(x):
         return head(inner(x))

      return chain(tail, wrap) if tail else wrap

   @nb.njit
   def add(x):
      return x + 1.2

   @nb.njit
   def mul(x):
      return x * 2

   # must be used outside of the jit
   addmul = chain((add, mul))
   addmuladd = chain((add, mul, add))

   @nb.njit
   def jit_with_chain_funcs():
      return addmul(3), addmuladd(3)

   assert jit_with_chain_funcs() == (7.2, 9.6)

@only_if_jit
def test_numba_chain_funcs_common_arg():
   @nb.njit
   def ident(x, arg):
      return x

   def chain(fs, inner=ident):
      head, tail = fs[-1], fs[:-1]

      @nb.njit
      def wrap(x, arg):
         return head(inner(x, arg), arg)

      return chain(tail, wrap) if tail else wrap

   @nb.njit
   def add(x, arg):
      return x + arg[0]

   @nb.njit
   def mul(x, arg):
      return x * arg[1]

   # must be used outside of the jit
   addmul = chain((add, mul))
   addmuladd = chain((add, mul, add))

   arg = (1.0, 2.0)

   @nb.njit
   def jit_with_chain_funcs():
      return addmul(3, arg), addmuladd(3, arg)

   assert jit_with_chain_funcs() == (7, 9)

@only_if_jit
def test_numba_chain_funcs_args_per():
   @nb.njit
   def ident(x, _):
      print("ident")
      return x

   def chain(fs, inner=ident):
      head, tail = fs[0], fs[1:]

      @nb.njit
      def wrap(x, argstack):
         return head(inner(x, argstack[:-1]), argstack[-1])  # why order different?!?

      return chain(tail, wrap) if tail else wrap

   @nb.njit
   def add(x, arg):
      if arg is 1:
         print("add 1")
      elif arg is 2:
         print("add 2")
      elif arg is 3:
         print("add 3")
      else:
         print("add ?")
      return x + arg

   @nb.njit
   def mul(x, arg):
      if arg is 1:
         print("mul 1")
      elif arg is 2:
         print("mul 2")
      elif arg is 3:
         print("mul 3")
      else:
         print("mul ?")
      return x * arg

   # must be used outside of the jit
   addmul = chain((mul, add))
   addmuladd = chain((add, mul, add))

   @nb.njit
   def jit_with_chain_funcs():
      x = np.array([3.0, 4.0])
      print("addmul")
      a = addmul(x, argstack=(1.1, 1))
      print("addmuladd")
      b = addmuladd(x, argstack=(3, 1.2, 1))
      print("done")
      return (a, b)

   tup = jit_with_chain_funcs()
   assert np.allclose(tup[0][0], (3 * 1.1) + 1)
   assert np.allclose(tup[1][0], ((3 + 3) * 1.2) + 1)
   assert np.allclose(tup[0][1], (4 * 1.1) + 1)
   assert np.allclose(tup[1][1], ((4 + 3) * 1.2) + 1)

@only_if_jit
def test_numba_cannot_chain_jitclass():
   @nb.experimental.jitclass((("member", nt.int32), ))
   class Ident:
      def __init__(self):
         pass

      def call(self, x):
         return x

   def chain(fs, inner=Ident()):
      head, tail = fs[-1], fs[:-1]

      @nb.experimental.jitclass((("dummy", nt.int32), ))
      class Wrap:
         def __init__(self):
            pass

         def call(self, x):
            return head.call(inner.call(x))  # no go

      return chain(tail, Wrap()) if tail else Wrap()

   @nb.experimental.jitclass((("member", nt.int32), ))
   class Foo:
      def __init__(self):
         pass

      def call(self, x):
         return x + 1.2

   @nb.experimental.jitclass((("member", nt.int32), ))
   class Bar:
      def __init__(self):
         pass

      def call(self, x):
         return x * 2

   # must be used outside of the jit
   addmul = chain((Foo(), Bar()))
   addmuladd = chain((Foo(), Bar(), Foo()))

   @nb.njit
   def jit_with_chain_funcs(a, b):
      return a.call(3), b.call(3)

   with pytest.raises(nb.TypingError):
      assert jit_with_chain_funcs(addmul, addmuladd) == (7.2, 9.6)

@only_if_jit
def test_numba_outargs():
   @nb.njit
   def foo(a):
      a[0] = 7

   a = np.arange(4)
   foo(a)
   assert a[0] == 7

@only_if_jit
def test_reshape():
   @nb.njit
   def numba_reshape(x):
      return x.reshape(-1, 4, 4)

   assert numba_reshape(np.arange(128)).shape == (8, 4, 4)
   assert numba_reshape(np.arange(128).reshape(4, 4, 4, 2)).shape == (8, 4, 4)

@pytest.mark.skip
@only_if_jit
def test_gil_release():
   @nb.njit(nogil=True)
   def sqrt_i_n(a, i, n):
      for i in range(i, len(a), n):
         a[i] = np.sqrt(a[i])

   x = np.arange(1000000)
   print(dir(sqrt_i_n))

   assert 0
