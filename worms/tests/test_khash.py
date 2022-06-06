import pytest
from worms.util import jit
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_get

def main():
   test_khash_jitclass()
   test_khash_numba_closure()
   test_khash()
   test_khash_closure()

# @pytest.mark.skip()
def test_khash_jitclass():
   @jit
   def use_khash_jitclass(h, i):
      return h.get(i) + 10

   h = KHashi8i8()
   h.update([(7, 3), (13, 4)])
   h.set(1, 13)
   assert h.get(1) == 13
   assert h.get(7) == 3
   assert h.get(13) == 4
   assert use_khash_jitclass(h, 1) == 23
   assert use_khash_jitclass(h, 7) == 13
   assert use_khash_jitclass(h, 13) == 14
   assert h.get(-2345) == -123456789
   assert h.get(926347) == -123456789
   assert h.size() == 3

# @pytest.mark.skip()
def test_khash_numba_closure():
   def foo(h):
      hash = h.hash

      @jit
      def func(i):
         return _khash_get(hash, i, -123456789)

      return func

   def numba_get(h, i):
      return h.get(i)

   h = KHashi8i8()
   h.set(10, 10)

   assert numba_get(h, 10) == 10

   f = foo(h)
   assert f(10) == 10

def test_khash():
   def use_khash(h, i):
      return h.get(i) + 10

   h = KHashi8i8()
   h.update([(7, 3), (13, 4)])
   h.set(1, 13)
   assert h.get(1) == 13
   assert h.get(7) == 3
   assert h.get(13) == 4
   assert use_khash(h, 1) == 23
   assert use_khash(h, 7) == 13
   assert use_khash(h, 13) == 14
   assert h.get(-2345) == -123456789
   assert h.get(926347) == -123456789
   assert h.size() == 3

def test_khash_closure():
   def get(h, i):
      return h.get(i)

   h = KHashi8i8()
   h.set(10, 10)
   assert get(h, 10) == 10

if __name__ == '__main__':
   main()
