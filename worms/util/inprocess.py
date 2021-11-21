'''
some multiprocessing related utils
InProcessExecutor, which tries to be sortof a dummy executor to ease debugging
'''

import multiprocessing, threading, os
from time import perf_counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed

def run_and_time(func, *args, **kw):
   t = perf_counter()
   return func(*args, **kw), perf_counter() - t

class InProcessExecutor:
   def __init__(self, *args, **kw):
      pass

   def __enter__(self):
      return self

   def __exit__(self, *args):
      pass

   def submit(self, fn, *args, **kw):
      return NonFuture(fn, *args, **kw)

   # def map(self, func, *iterables):
   # return map(func, *iterables)
   # return (NonFuture(func(*args) for args in zip(iterables)))

class NonFuture:
   def __init__(self, fn, *args, dummy=None, **kw):
      self.fn = fn
      self.dummy = not callable(fn) if dummy is None else dummy
      self.args = args
      self.kw = kw
      self._condition = threading.Condition()
      self._state = "FINISHED"
      self._waiters = []

   def result(self):
      if self.dummy:
         return self.fn
         print('NonFuture running')
      rslt = self.fn(*self.args, **self.kw)
      # print('result obj\n  ', rslt)
      # assert 0
      return rslt

def cpu_count():
   try:
      return int(os.environ["SLURM_CPUS_ON_NODE"])
   except:
      return multiprocessing.cpu_count()

def parallel_batch_map(pool, function, accumulator, batch_size, map_func_args, **kw):
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   os.environ["NUMEXPR_NUM_THREADS"] = "1"
   njobs = len(map_func_args[0])
   args = list(zip(*map_func_args))
   for ibatch in range(0, njobs, batch_size):
      beg = ibatch
      end = min(njobs, ibatch + batch_size)
      batch_args = args[beg:end]  # todo, this could be done lazily...
      futures = [pool.submit(function, *a) for a in batch_args]
      if isinstance(pool, (ProcessPoolExecutor, ThreadPoolExecutor)):
         as_completed = cf_as_completed
      elif isinstance(pool, InProcessExecutor):
         as_completed = lambda x: x
      else:
         from dask.distributed import as_completed as dd_as_completed

         as_completed = dd_as_completed
      for _ in accumulator.accumulate(as_completed(futures)):
         yield None
      accumulator.checkpoint()

def parallel_nobatch_map(pool, function, accumulator, batch_size, map_func_args, **kw):
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   os.environ["NUMEXPR_NUM_THREADS"] = "1"
   njobs = len(map_func_args[0])
   args = list(zip(*map_func_args))
   futures = [pool.submit(function, *a) for a in args]
   if isinstance(pool, (ProcessPoolExecutor, ThreadPoolExecutor)):
      as_completed = cf_as_completed
   else:
      as_completed = dd_as_completed
   for _ in accumulator.accumulate(as_completed(futures)):
      yield None
   accumulator.checkpoint()

def tqdm_parallel_map(pool, function, accumulator, map_func_args, batch_size, **kw):
   for _ in tqdm(
         parallel_batch_map(pool, function, accumulator, batch_size, map_func_args=map_func_args,
                            **kw), total=len(map_func_args[0]), **kw):
      pass
