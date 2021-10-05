try:
   import os
   if 'NUMBA_DISABLE_JIT' in os.environ:
      raise ImportError
   import numba
   from numba.types import float64, float32, int64, int32
   jit = numba.njit(nogil=True, fastmath=True)

   def guvec(sigs, layout, func):
      return numba.guvectorize(sigs, layout, nopython=True,
                               fastmath=True)(func)  # nogil not supported

except ImportError:
   import numpy
   # dummy
   float64 = float32 = int64 = int32 = numpy.empty((1, 1, 1, 1, 1, 1, 1))
   jit = lambda f: None

   def guvec(sigs, layout, func):
      return None