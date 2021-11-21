import sys, os, json, shutil, collections, tarfile, tempfile, itertools
from collections.abc import Iterable
import worms

def print_collisions():
   olap = collections.defaultdict(list)
   for e in alldb:
      olap[os.path.basename(e['file'])].append(e['file'])
   for k, v in olap.items():
      if len(v) > 1:
         print('archive collision!', k)
         for dup in v:
            print('   dup:', dictdb[dup])
         print()

def archive_fname(fname):
   return fname.replace('/', r'\\')

class Archive:
   def __init__(self, dbname, bblocks, pdbs, metadata, **kw):
      self.dbname = dbname
      self.bblocks = bblocks
      self.pdbs = pdbs
      self.metadata = metadata
      assert len(pdbs) == len(bblocks)

   def __len__(self):
      return len(self.bblocks)

   def __str__(self):
      return os.sep.join(dbname, self.metadata)

def make_bblock_archive(
      dbfiles=list(),
      dbcontents=list(),
      target='localdb',
      dbname=None,
      nbblocks=9e9,
      overwrite=True,
      extrafiles=list(),
      pdb_contents=dict(),
):
   'produce an lzma tarball with one json file and all the pdbs referenced'

   infnames = list()
   if dbfiles:
      assert not dbcontents
      if isinstance(dbfiles, str):
         dbfiles = [dbfiles]
      rbd = worms.database.read_bblock_dbfiles(dbfiles)
      dbcontents, _, _, pdb_contents2 = rbd
      pdb_contents.update(pdb_contents2)
      assert len(infnames) == len(set(infnames))

   # print('make_bblock_archive file entries')
   # for e in dbcontents:
   # print('  ', e['file'])

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
   if os.path.exists(fname) and not overwrite:
      print('file exists:', fname)
      sys.exit(-1)

   with tarfile.open(fname + '.tmp', 'x:xz') as tarball:
      tmpfile = tempfile.mkstemp()[1]
      with open(tmpfile, 'w') as out:
         out.write('dbname: ' + dbname + os.linesep)
         out.write('targetfile: ' + target + os.linesep)
         out.write('timestamp: ' + worms.util.datetimetag() + os.linesep)
         assert len(set(tarball.getnames())) == len(tarball.getnames())
      tarball.add(tmpfile, 'metadata.txt')

      seenit = dict()
      for i, e in enumerate(dbcontents):
         if i % 100 == 0:
            print(f'    progress {int(i / len(dbcontents) * 100)   }%')
         f = e['file']
         if f in pdb_contents:
            tmppdb = tempfile.mkstemp()[1]
            with open(tmppdb, 'w') as out:
               out.write(pdb_contents[f])
            newf = os.path.basename(f)
            origf = f
            f = tmppdb
         elif os.path.exists(f):
            origf = f
            newf = os.path.join('pdbs', archive_fname(f))
         else:
            assert os.path.exists(
               f) or f in pdb_contents, 'file does not exist and is not in database pdb_contents'

         # assert not newf in seenit
         if newf in seenit:
            print('warning, skipping duplicate destination pdb file')
            assert seenit[newf] in pdb_contents or origf == seenit[newf]
            continue
         seenit[newf] = origf

         print('tarball.add')
         print('   FILE', f)
         print('   ARCNAME', newf)
         tarball.add(f, newf)
         e['file'] = newf
      tmpfile = tempfile.mkstemp()[1]
      with open(tmpfile, 'w') as out:
         json.dump(dbcontents, out, indent=2)
      newfname = os.path.join(dbname + '.json')
      tarball.add(tmpfile, newfname)

      for f in map(os.path.abspath, extrafiles):
         newfname = os.path.join('extras', archive_fname(f))
         print('tarball.add EXTRA')
         print('   FILE', f)
         print('   ARCNAME', newfname)
         tarball.add(f, newfname)
      for f in map(os.path.abspath, dbfiles):
         newfname = os.path.join('sources', archive_fname(f))
         print('tarball.add SOURCE')
         print('   FILE', f)
         print('   ARCNAME', newfname)
         tarball.add(f, newfname)

   os.rename(fname + '.tmp', fname)

   return fname

def read_bblock_archive(fname):
   '''read bblock archive(s)'''
   if isinstance(fname, (str, )):
      return _read_bblock_archive_one(fname)
   else:
      return _read_bblock_archive_many(fname)

def _read_bblock_archive_one(fname):
   'read json and pdb contents from tarball'
   print('read archive', fname)

   with tempfile.TemporaryDirectory() as tmpdir:
      if not os.path.exists(fname) and fname.count('/') == 0:
         fname = worms.data.get_database_archive_path(fname)
      with tarfile.open(fname, 'r:xz') as inp:
         names = inp.getnames()
         inp.extractall(tmpdir)
      assert len(set(names)) == len(names), 'duplicate names in tarfile'

      dbname = '.'.join(os.path.basename(fname).split('.')[:-1])
      print('dbname', dbname)
      bblocks = list()
      metadata = None
      pdbs = dict()
      # print('arc contents')
      # for fn in names:
      #    print('   ', fn)
      for fn in names:
         if fn == 'metadata.txt':
            print(f'{    f" METADATA "    :*^80}')
            print(f'{    f" {dbname} "    :*^80}')
            with open(os.path.join(tmpdir, fn)) as inp:
               metadata = inp.read()
               print(metadata)
            print('*' * 80)
         elif fn.startswith('sources/'):
            # print('arcive source file', fn)
            pass
         elif fn.startswith('extras/'):
            # print('arcive extra file', fn)
            pass
         elif fn.endswith(('.json', '.txt')):
            print('arcive reading json file', fn)
            assert not bblocks, 'should be only one non-extra json file in archive'
            with open(os.path.join(tmpdir, fn), 'r') as inp:
               s = inp.read()
               bblocks.extend(json.loads(s))
         elif fn.endswith('.pdb'):
            # print('arcive reading pdb file', len(pdbs), fn)
            with open(os.path.join(tmpdir, fn)) as inp:
               pdbs[fn] = inp.read()

         else:
            raise ValueError('don\'t know what to do with file %s', fn)
   assert dbname is not None
   assert bblocks is not None
   assert len(pdbs)
   from collections import Counter
   filecount = Counter([e['file'] for e in bblocks])
   for k, v in filecount.items():
      if v > 1:
         print('duplicate')
         print('   ', k)
         assert 0, 'duplicate found'

   # print('db files')
   # for k in filecount:
   #    print('  ', k)

   pdbkey = list(sorted(pdbs.keys()))
   files = list(sorted(e['file'] for e in bblocks))
   if not pdbkey == files:
      # print('mismatch in archive')
      # for k in sorted(pdbs.keys()):
      #    print('   key', k)
      # print('files:')
      # for f in sorted(e['file'] for e in bblocks):
      #    print('   file', f)
      assert 0, 'pbdkeys and fnames mismatch'

   return Archive(dbname, bblocks, pdbs, metadata)

def _read_bblock_archive_many(fnames, dbname=None):
   'read json and pdb contents from tarball, merging multiple archives'
   mapped = list(map(_read_bblock_archive_one, fnames))
   names0 = [x.dbname for x in mapped]
   bblocks0 = [bblocks for x in mapped]
   pdb_contents0 = [x.pdbs for x in mapped]
   metadata0 = [x.metadata for x in mapped]

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

   return Archive(dbname, bblocks, pdb_contents, metadata0)

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
