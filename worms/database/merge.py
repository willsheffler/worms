import json, itertools, re, collections

from numpy.lib.arraysetops import isin

import worms
from worms.database.archive import read_bblock_archive

def fields_match(stuff, field):
   return [stuff[0][field]] * len(stuff) == [d[field] for d in stuff]

def formatjson(stuff):
   s = json.dumps(stuff, indent=3)
   s = s.replace('"', '')
   # s = re.sub(r' \[\n', '\n', s)
   # s = re.sub(r' *\],\n', '', s)
   return s

def merge_json_databases(
      jsondbs,
      dump_archive='',
      overwrite=False,
      pdb_contents=dict(),
):

   extra = list()
   if isinstance(jsondbs[0], str):
      assert all(isinstance(x, str) for x in jsondbs)
      jsonfnames = jsondbs
      jsondbs = list()
      for f in jsonfnames:
         assert f.endswith(('json', '.txt'))
         with open(f) as inp:
            jsondbs.append(json.load(inp))
         extra.append(f)

   entries = collections.defaultdict(list)
   assert isinstance(jsondbs[0][0], dict)
   for e in itertools.chain(*jsondbs):
      entries[e['file']].append(e)
   # for k, v in entries.items():
   # print(len(v), k)

   newdb = list()
   for fname, dups in entries.items():
      # make sure all fields match, except connections
      for field in 'name class type base components validated protocol'.split():
         assert fields_match(dups, field)
      mergeconn = collections.defaultdict(list)
      for dup in dups:
         for c in dup['connections']:
            # print(c)
            k = (c['chain'], c['direction'])
            for r in c['residues']:
               if isinstance(r, int):
                  mergeconn[k].append([r])
               elif isinstance(r, str):
                  lb, ub = map(int, r.split(':'))
                  if lb > ub: lb, ub = ub, lb
                  mergeconn[k].append(range(lb, ub + 1))
               else:
                  print('cant interpret res entry', c[''])
                  print('cant interpret res entry', r)
                  assert 0

      for (c, d), r in mergeconn.items():
         merged = sorted(set(itertools.chain(*r)))

         # todo: merge "chain,range" types also
         mergeconn[c, d] = merged
      newentry = dups[0]  # take info from first duplicate TODO check fields match?
      newentry['connections'].clear()
      for (c, d), r in mergeconn.items():
         newentry['connections'].append(dict(
            chain=c,
            direction=d,
            residues=r,
         ))
      newdb.append(newentry)
   if dump_archive:
      dbfname = worms.database.archive.make_bblock_archive(
         dbcontents=newdb,
         target=dump_archive,
         nbblocks=9e9,
         overwrite=overwrite,
         extrafiles=extra,
         pdb_contents=pdb_contents,
      )
   return newdb

def merge_main(archivename, dbfiles):
   jsondbs = list()
   pdb_contents = dict()
   assert archivename.endswith('.txz')
   for f in dbfiles:
      if f.endswith('.txz'):
         arc = read_bblock_archive(f)
         jsondbs.append(arc.bblocks)
         pdb_contents.update(arc.pdbs)
      else:
         print('"' + f + '"')
         assert f.endswith(('json', '.txt'))
         with open(f) as inp:
            jsondbs.append(json.load(inp))

   merge_json_databases(
      jsondbs,
      dump_archive=archivename,
      overwrite=True,
      pdb_contents=pdb_contents,
   )
