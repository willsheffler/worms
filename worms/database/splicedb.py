import os, pickle
import worms

class SpliceDB:
   """Stores valid NC splices for bblock pairs"""
   def __init__(self, **kw):
      self._cache = dict()

   def partial(self, params, pdbkey):
      assert isinstance(pdbkey, int)
      if (params, pdbkey) not in self._cache:
         self._cache[params, pdbkey] = dict()
      return self._cache[params, pdbkey]

   def has(self, params, pdbkey0, pdbkey1):
      k = (params, pdbkey0)
      if k in self._cache:
         if pdbkey1 in self._cache[k]:
            return True
      return False

   def add(self, params, pdbkey0, pdbkey1, val):
      assert isinstance(pdbkey0, int)
      assert isinstance(pdbkey1, int)
      k = (params, pdbkey0)
      if k not in self._cache:
         self._cache[k] = dict()
      self._cache[k][pdbkey1] = val

   # def cachepath(self, params, pdbkey):
   #    # stock hash ok for tuples of numbers (?)
   #    prm = "%016x" % abs(hash(params))
   #    key = "%016x.pickle" % pdbkey
   #    for d in self.cachedirs:
   #       candidate = os.path.join(d, prm, key)
   #       if os.path.exists(candidate):
   #          return candidate
   #    return os.path.join(self.cachedirs[0], prm, key)

   def sync_to_disk(self, *_):
      pass

   def listpath(self, params, pdbkey):
      return ''

   def clear(self):
      # do nothing, as can't reload from cache
      pass

   def merge_into_self(self, other):
      keysself = set(self._cache.keys())
      keysother = set(other._cache.keys())
      self._cache.update({k: other._cache[k] for k in keysother - keysself})
      for k in keysself.intersection(keysother):
         vself = self._cache[k]
         vother = other._cache[k]
         vself.update(vother)

   def __str__(self):
      degree = [len(v) for v in self._cache]
      return os.linesep.join([
         f'SpliceDB',
         f'   num lhs: {len(self._cache)}',
         f'   npairs: {sum(degree)}',
         f'   num rhs: {degree}',
      ])

class CachingSpliceDB:
   """Stores valid NC splices for bblock pairs"""
   def __init__(self, cachedirs=None, **kw):
      cachedirs = worms.database.get_cachedirs(cachedirs)
      self.cachedirs = [os.path.join(x, "splices") for x in cachedirs]
      print("CachingSpliceDB cachedirs:", self.cachedirs)
      self._cache = dict()
      self._dirty = set()

   def partial(self, params, pdbkey):
      assert isinstance(pdbkey, int)
      if (params, pdbkey) not in self._cache:
         cachefile = self.cachepath(params, pdbkey)
         if os.path.exists(cachefile):
            with open(cachefile, "rb") as f:
               self._cache[params, pdbkey] = pickle.load(f)
         else:
            self._cache[params, pdbkey] = dict()

      return self._cache[params, pdbkey]

   def has(self, params, pdbkey0, pdbkey1):
      k = (params, pdbkey0)
      if k in self._cache:
         if pdbkey1 in self._cache[k]:
            return True
      return False

   def add(self, params, pdbkey0, pdbkey1, val):
      assert isinstance(pdbkey0, int)
      assert isinstance(pdbkey1, int)
      k = (params, pdbkey0)
      self._dirty.add(k)
      if k not in self._cache:
         self._cache[k] = dict()
      self._cache[k][pdbkey1] = val

   def cachepath(self, params, pdbkey):
      # stock hash ok for tuples of numbers (?)
      prm = "%016x" % abs(hash(params))
      key = "%016x.pickle" % pdbkey
      for d in self.cachedirs:
         candidate = os.path.join(d, prm, key)
         if os.path.exists(candidate):
            return candidate
      return os.path.join(self.cachedirs[0], prm, key)

   def listpath(self, params, pdbkey):
      return self.cachepath(params, pdbkey).replace(".pickle", "_list.pickle")

   def sync_to_disk(self, dirty_only=True):
      worms.PING('CachingSpliceDB sync_to_disk')
      for i in range(10):
         keys = list(self._dirty) if dirty_only else self.cache.keys()
         for key in keys:
            cachefile = self.cachepath(*key)
            if os.path.exists(cachefile + ".lock"):
               continue
            os.makedirs(os.path.dirname(cachefile), exist_ok=True)
            with open(cachefile + ".lock", "w"):
               if os.path.exists(cachefile):
                  with open(cachefile, "rb") as inp:
                     self._cache[key].update(pickle.load(inp))
               with open(cachefile, "wb") as out:
                  data = self._cache[key]
                  pickle.dump(data, out)
               listfile = self.listpath(*key)
               with open(listfile, "wb") as out:
                  data = set(self._cache[key].keys())
                  pickle.dump(data, out)
            os.remove(cachefile + ".lock")
            self._dirty.remove(key)
      if len(self._dirty):
         print(self._dirty)
         print("warning: some caches unsaved", len(self._dirty))
      worms.PING('CachingSpliceDB sync_to_disk DONE')

   def clear(self):
      self._cache.clear()

   def __str__(self):
      degree = [len(v) for v in self._cache]
      return os.linesep.join([
         f'SpliceDB',
         f'   num lhs: {len(self._cache)}',
         f'   npairs: {sum(degree)}',
         f'   num rhs: {degree}',
      ])
