import os, pickle, lzma, json
import worms

data_dir = os.path.dirname(__file__)
pdb_dir = os.path.join(data_dir, 'pdb')
database_dir = os.path.join(data_dir, 'databases')
json_dir = os.path.join(data_dir, 'databases', 'json')

def data_file_path(fname):
   assert not fname.startswith('/')
   return os.path.join(data_dir, fname)

def test_file_path(fname):
   assert not fname.startswith('/')
   return os.path.join(data_dir, 'test_cases', fname)

def data_file_contents(fname, mode='r'):
   fname = data_file_path(fname)
   with open(fname, mode) as inp:
      return inp.read()

def test_file_contents(fname, mode='r'):
   fname = test_file_path(fname)
   with open(fname, mode) as inp:
      return inp.read()

def test_file_json(fname, mode='r'):
   fname = test_file_path(fname)
   return load_json(fname)

def get_database_archive_path(dbname):
   return os.path.join(database_dir, dbname)

def get_testing_database(dbname):
   return worms.load(os.path.join(data_dir, 'databases', dbname + '.pickle.xz'))

def save_testing_database(db, dbname):
   return worms.dump(db, os.path.join(data_dir, 'databases', dbname + '.pickle.xz'))

def load_json(f):
   with open(f, 'r') as inp:
      return json.load(inp)

def dump_json(j, f, indent=True):
   with open(f, 'w') as out:
      return json.dump(j, out, indent=indent)

def load(fname, add_dotpickle=True, assume_lzma=True):
   opener = open
   if fname.endswith('.xz'):
      opener = open_lzma_cached
   elif not fname.endswith('.pickle'):
      if assume_lzma:
         opener = open_lzma_cached
         fname += '.pickle.xz'
      else:
         fname += 'pickle'
   with opener(fname) as inp:
      return pickle.load(inp)

def dump(stuff, fname, add_dotpickle=True, uselzma=True):
   opener = open
   if fname.endswith('.xz'):
      assert fname.endswith('.pickle.xz')
      opener = lzma.open
   elif uselzma:
      opener = lzma.open
      if not fname.endswith('.pickle'):
         fname += '.pickle'
      fname += '.xz'
   with opener(fname, 'wb') as out:
      pickle.dump(stuff, out)

class open_lzma_cached:
   def __init__(self, fname):
      fname = os.path.abspath(fname)
      if not os.path.exists(fname + '.decompressed'):
         xzfile = lzma.open(fname, 'rb')
         with open(fname + '.decompressed', 'wb') as out:
            out.write(xzfile.read())
      else:
         self.file_obj = open(fname + '.decompressed', 'rb')

   def __enter__(self):
      return self.file_obj

   def __exit__(self, __type__, __value__, __traceback__):
      self.file_obj.close()
