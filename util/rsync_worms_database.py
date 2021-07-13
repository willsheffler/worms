import os, json, argparse, re
from tqdm import tqdm

parser = argparse.ArgumentParser(description="move worms db around")
parser.add_argument("locations", nargs="+", type=str)
parser.add_argument("--destination", default="./dbfiles", type=str)
args = parser.parse_args()
dest = args.destination + '/'

for dbfile in args.locations:
   os.makedirs(dest, exist_ok=1)
   with open(dbfile) as inp:
      dbcontents = inp.read()
   dbjson = json.loads(dbcontents)
   files = [e["file"] for e in dbjson]
   for f in tqdm(files):
      newf = re.sub('/.*/', dest, f)
      entry = [e for e in dbjson if e['file'] == f]
      assert len(entry) == 1
      entry[0]['file'] = newf
      cmd = f"rsync -z {f} {newf}"
      # print(cmd)
      os.system(cmd)
   newdbfile = dest + os.path.split(dbfile)[1]
   print('reading from', dbfile)
   print('moving to', newdbfile)
   with open(newdbfile, 'w') as out:
      json.dump(dbjson, out)
