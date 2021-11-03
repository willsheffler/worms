import os, json, shutil, collections, tarfile, tempfile, itertools
from collections.abc import Iterable
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

def archive_fname(fname):
   return fname.replace('/', '\\')

def make_bblock_archive(
   dbfiles=None,
   dbcontents=None,
   target='localdb',
   dbname=None,
   nbblocks=9e9,
   overwrite=False,
):
   'produce an lzma tarball with one json file and all the pdbs referenced'

   if dbfiles:
      assert not dbcontents
      if isinstance(dbfiles, str):
         dbfiles = [dbfiles]
      dbcontents, _, _ = worms.database.read_bblock_dbfiles(dbfiles)
      fnames = [e['file'] for e in dbcontents]
      assert len(fnames) == len(set(fnames))

   if target.endswith('.txz'):
      target = target[:-4]
   if target.endswith('.tar.xz'):
      target = target[:-7]
   if dbname is None:
      dbname = os.path.basename(target)

   if os.path.dirname(target) != '':
      os.makedirs(os.path.dirname(target), exist_ok=True)
   mode = 'w:xz' if overwrite else 'x:xz'
   fname = target + '.txz'
   with tarfile.open(target + '.txz', mode) as tarball:
      for i, e in enumerate(dbcontents):
         if i % 100 == 0: print(f'    progress {int(i / len(dbcontents) * 100)   }%')
         f = e['file']
         newf = os.sep.join([dbname, archive_fname(f)])

         tarball.add(f, newf)
         e['file'] = newf
         tmpfile = tempfile.mkstemp()[1]
      with open(tmpfile, 'w') as out:
         json.dump(dbcontents, out, indent=2)
      tarball.add(tmpfile, os.sep.join([dbname, dbname + '.json']))
      assert len(set(tarball.getnames())) == len(tarball.getnames())
   return fname

def read_bblock_archive(fname):
   '''read bblock archive(s)'''
   if isinstance(fname, (str, )):
      return _read_bblock_archive_one(fname)
   else:
      return _read_bblock_archive_many(fname)

def _read_bblock_archive_one(fname):
   'read json and pdb contents from tarball'
   with tempfile.TemporaryDirectory() as tmpdir:
      if not os.path.exists(fname) and fname.count('/') == 0:
         fname = worms.data.get_database_archive_path(fname)
      with tarfile.open(fname, 'r:xz') as inp:
         names = inp.getnames()
         inp.extractall(tmpdir)
      assert len(set(names)) == len(names), 'duplicate names in tarfile'

      dbname = None
      bblocks = None
      pdbs = dict()
      for fn in names:
         s = fn.split(os.sep)
         if len(s) == 1:  # just a directory could be included
            assert dbname is None
            dbname = s[0]
         else:
            if dbname is None:
               dbname = s[0]
            assert s[0] == dbname
            if fn.endswith('.json'):
               assert bblocks is None
               with open(os.sep.join([tmpdir, fn]), 'r') as inp:
                  bblocks = json.load(inp)
            else:
               assert fn.endswith('.pdb')
               with open(os.sep.join([tmpdir, fn])) as inp:
                  pdbs[fn] = inp.read()

   assert dbname is not None
   assert bblocks is not None
   assert len(pdbs)
   assert list(sorted(pdbs.keys())) == list(sorted(e['file'] for e in bblocks))

   return dbname, bblocks, pdbs

def _read_bblock_archive_many(fnames, dbname=None):
   'read json and pdb contents from tarball, merging multiple archives'
   mapped = list(map(_read_bblock_archive_one, fnames))
   names0 = [x[0] for x in mapped]
   bblocks0 = [x[1] for x in mapped]
   pdb_contents0 = [x[2] for x in mapped]

   dbname = '/'.join(names0)
   bblocks = list(itertools.chain(*bblocks0))
   pdb_contents = dict()
   for d in pdb_contents0:
      pdb_contents.update(d)
   ndups = len(bblocks) - len(pdb_contents)
   if ndups > 0:
      print(f'WARNING {ndups} ({int(ndups/len(bblocks)*100)}%) duplicate pdbs reading archives:')
      for i, n in enumerate(fnames):
         print(f' {len(bblocks0[i]):5} {n}')

   return dbname, bblocks, pdb_contents

# def make_bblockdb(fname):

if __name__ == '__main__':
   # dbfiles = [
   # '/home/swang523/Crystal_engineering/worms/8-16-21_cage_xtal/input/selected_good.json'
   # ]
   # fname = make_bblock_archive(dbfiles, '/home/sheffler/tmp/cagextal_selected_good',
   # overwrite=True)

   dbarc = '/home/sheffler/tmp/cagextal_selected_good.txz'

   t = worms.Timer().start()

   name, bbdb, pdb_contents = read_bblock_archive(dbarc)
   print(name, len(bbdb), sum(len(v) for v in pdb_contents.values()) / 1000_000, 'M')
   # for k, v in pdb_contents.items():
   # print(len(v), k)
   t.checkpoint('read archive')
   # assert 0

   name, bbdb, pdb_contents = read_bblock_archive([dbarc, dbarc, dbarc])
   print(name, len(bbdb), sum(len(v) for v in pdb_contents.values()) / 1000_000, 'M')
   # for k, v in pdb_contents.items():
   # print(len(v), k)
   t.checkpoint('read multiple')

   print(t.report())
