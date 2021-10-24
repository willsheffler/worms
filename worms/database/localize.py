import os, json, shutil, collections, tarfile, tempfile
import worms

def print_collisions():
   olap = collections.defaultdict(list)
   for e in alldb:
      olap[os.path.basename(e['file'])].append(e['file'])
   for k, v in olap.items():
      if len(v) > 1:
         print(k)
         for dup in v:
            print(dictdb[dup])
         print()

def localize_fname(fname):
   return fname.replace('/', '\\')

def open_txz_database(fname):
   pass

def make_packed_bblockdb(dbfiles, target='localdb', dbname=None, nbblocks=9e9, overwrite=False):
   if target.endswith('.txz'):
      target = target[:-4]
   if target.endswith('.tar.xz'):
      target = target[:-7]
   if dbname is None:
      dbname = os.path.basename(target)

   alldb, dictdb, k2pdb = worms.database.read_bblock_dbfiles(dbfiles)
   os.makedirs(os.path.dirname(target), exist_ok=True)
   mode = 'w:xz' if overwrite else 'x:xz'
   with tarfile.open(target + '.txz', mode) as tarball:
      for i, e in enumerate(alldb):
         f = e['file']
         newf = os.path.join(dbname, localize_fname(f))
         print(f'copying {int(i / len(alldb) * 100)   }% {newf}')
         # shutil.copyfile(f, os.path.join(target, newf))
         tarball.add(f, newf)
         e['file'] = newf
         tmpfile = tempfile.mkstemp()[1]
         with open(tmpfile, 'w') as out:
            json.dump(alldb, out, indent=2)
         tarball.add(tmpfile, os.path.join(dbname, dbname + '.json'))

   # fnames = {e['file'] for e in alldb}
   # basenames = {os.path.basename(e['file']) for e in alldb}
   # assert len(alldb) == len(fnames)
   # assert len(alldb) == len(basenames)

if __name__ == '__main__':
   dbfiles = [
      '/home/swang523/Crystal_engineering/worms/8-16-21_cage_xtal/input/selected_good.json'
   ]
   # make_packed_bblockdb(dbfiles, '/home/sheffler/tmp/cagextal_selected_good', overwrite=True)

   with tarfile.open('/home/sheffler/tmp/cagextal_selected_good.txz') as inp:
      db = json.load(inp.extractfile('cagextal_selected_good/cagextal_selected_good.json'))

      # print(inp.getmembers()

      strs = {name: inp.extractfile(name).read() for name in inp.getnames()}

      # import pyrosetta
      # pyrosetta.init()
      # pose = pyrosetta.Pose()
      # pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb1)
      # pose.dump_pdb('/home/sheffler/tmp/cagextal_selected_good/foo.pdb')

   print(len(strs))
   for k, v in strs.items():
      print(len(v), k)
