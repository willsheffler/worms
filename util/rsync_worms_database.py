import os, json, argparse, re, random
from tqdm import tqdm

def main():
   parser = argparse.ArgumentParser(description='move worms db around')
   parser.add_argument('--locations', default=[], nargs='+', type=str)
   parser.add_argument('--destination', default='./dbfiles', type=str)
   parser.add_argument('--maxentries', default=-1, type=int)
   parser.add_argument('--shuffle', default=False, action='store_true')
   parser.add_argument('--copy_all', default=False, action='store_true')
   args = vars(parser.parse_args())

   if args['copy_all']:
      copy_all(**args)
   else:
      copy_databases(**args)

def copy_databases(locations, destination, maxentries, shuffle, **kw):
   dest = destination + '/'
   for dbfile in locations:
      os.makedirs(dest, exist_ok=1)
      with open(dbfile) as inp:
         dbcontents = inp.read()
      dbjson = json.loads(dbcontents)
      newdbjson = list()
      files = [e['file'] for e in dbjson]
      random.shuffle(files)
      files = files[:maxentries]
      # print('=' * 80)
      # print(locations)
      # print('=' * 80)
      jobs = files
      # jobs = tqdm(files, miniters=10)
      for f in jobs:
         newf = re.sub('/.*/', dest, f)
         entry = [e for e in dbjson if e['file'] == f]
         assert len(entry) == 1
         entry = entry[0]
         entry['file'] = newf
         newdbjson.append(entry)
         cmd = f'rsync -z {f} {newf}'
         # print(cmd)
         os.system(cmd)
      newdbfile = dest + os.path.basename(dbfile).replace('.txt', '.json')
      print('creating from:', dbfile, flush=True)
      # print('moving to', newdbfile)
      with open(newdbfile, 'w') as out:
         json.dump(newdbjson, out, indent=4)

def copy_all(locations, destination, **kw):
   assert locations == []
   assert destination == './dbfiles'
   dest0 = '/home/sheffler/data/worms/databases/local'

   for path, namelist in zip(dbpaths, dbfiles):
      for name in namelist:
         locations = [path + '/' + name + '.txt']
         destination = dest0 + '/wormsdb_local_' + name
         copy_databases(locations, destination, **kw)

pathyang = '/home/yhsia/helixfuse/2018-07-09_sym/processing/database/'
dbyang = ['HFuse_Cx_database.20180711']
pathgeorge = '/home/yhsia/helixfuse/cyc_george/processing/database/'
dbgeorge = ['HFuse_Gcyc_database.20180817']
pathzibo = '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/'
dbzibo = [
   'HFuse_het_2chain_2arm_database.ZCON-103_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-112_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-127_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-131_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-13_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-15_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-34_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-37_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-39_2.20180516',
   'HFuse_het_2chain_2arm_database.ZCON-9_2.20180516',
]
pathsherry = '/home/yhsia/helixfuse/asym_sh_combine/processing/database/'
dbsherry = [
   'HFuse_het_3chain_2arm_database.Sh13_3.20180516',
   'HFuse_het_3chain_2arm_database.Sh29_3.20180516',
   'HFuse_het_3chain_2arm_database.Sh34_3.20180516',
   'HFuse_het_3chain_3arm_database.Sh13_3.20180516',
   'HFuse_het_3chain_3arm_database.Sh29_3.20180516',
   'HFuse_het_3chain_3arm_database.Sh34_3.20180516',
]
pathsherry2 = '/home/yhsia/helixfuse/2018-09-04_asym_sh_hetc2/combine/processing/database/'
dbsherry2 = [
   'HFuse_het_2chain_2arm_database.Sh13-5+1-AI_2.20180905',
   'HFuse_het_2chain_2arm_database.Sh29-5+1-CI_2.20180905'
]

dbpaths = [pathyang, pathgeorge, pathzibo, pathsherry, pathsherry2]
dbfiles = [dbyang, dbgeorge, dbzibo, dbsherry, dbsherry2]

if __name__ == '__main__':
   main()
   print('DONE', flush=True)
