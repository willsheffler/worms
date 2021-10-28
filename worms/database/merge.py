import json, itertools, re, collections

import worms

def fields_match(stuff, field):
   return [stuff[0][field]] * len(stuff) == [d[field] for d in stuff]

def formatjson(stuff):
   s = json.dumps(stuff, indent=3)
   s = s.replace('"', '')
   # s = re.sub(r' \[\n', '\n', s)
   # s = re.sub(r' *\],\n', '', s)
   return s

def merge_json_databases(jsondbs, dump_archive=''):
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
            if not isinstance(c['residues'][0], int):
               raise ValueError('cannot merge residue ranges')
            k = (c['chain'], c['direction'])
            mergeconn[k].append(c['residues'])
      for (c, d), r in mergeconn.items():
         merged = sorted(itertools.chain(*r))

         # todo: merge "chain,range" types also
         mergeconn[c, d] = merged
      newentry = dups[0]
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
         dbfiles=None,
         dbcontents=newdb,
         target=dump_archive,
         dbname=None,
         nbblocks=9e9,
         overwrite=False,
      )
   return newdb
