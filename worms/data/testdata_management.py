import os, pickle

from worms.util.util import datetimetag
from worms.util.ping import PING

from worms.data.data import data_dir, test_file_path

def get_latest_resulttables_path(tag, candidates_ok=False):
   path = get_timestamped_test_dir_latest(tag, candidates_ok=candidates_ok)
   if path is None:
      return None
   fname = os.path.join(path, 'reference_results.pickle')
   return fname

def get_latest_resulttables(tag, candidates_ok=False):
   fname = get_latest_resulttables_path(tag, candidates_ok=candidates_ok)
   try:
      with open(fname, 'rb') as inp:
         return fname, pickle.load(inp)
   except (TypeError, FileNotFoundError):
      return fname, None

def make_timestamped_test_dir(tag, candidate=True):
   timetag = datetimetag()
   testpath = test_file_path(tag)
   path = os.path.join(testpath, timetag)
   if candidate: path += '_CANDIDATE'
   os.makedirs(path, exist_ok=True)
   return path + '/'

def get_timestamped_test_dirs(tag, candidates_ok=False):
   testpath = test_file_path(tag)
   dirs = os.listdir(testpath)
   dirs = [f for f in dirs if os.path.isdir(os.path.join(testpath, f))]
   dirs = sorted(map(str, dirs))
   if not candidates_ok:
      dirs = [d for d in dirs if not d.endswith('_CANDIDATE')]
   try:
      dirs.remove('config')  # config directory allowed along with results dirs
   except ValueError:
      PING(f'WARNING no config dir for testdir {testpath}')
   return dirs

def get_timestamped_test_dir_latest(tag, candidates_ok=False):
   testpath = test_file_path(tag)
   dirs = get_timestamped_test_dirs(tag, candidates_ok=candidates_ok)
   if len(dirs) == 0:
      return None
   latest = dirs[-1]
   path = os.path.join(testpath, latest)
   assert os.path.isdir(path)
   return path + '/'
