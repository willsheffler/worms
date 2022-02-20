import sys, re

dbfiles = sys.argv[1:]

for dbfile in dbfiles:
   print(dbfile)
   with open(dbfile) as inp:
      contents = inp.read
      fnames = re.match('"file"\s*:\s*"?P<pdb>"', inp)
      print(fnames)

for i in $(grep file $dbfile | sed -e s=\{\"file\"\:\ \"==g | sed -e s=\",==g); do
   rsync -avz fw.bakerlab.org:$i ./dbfile_pdbs; 
done
grep file $dbfile | wc -l
ls dbfile_pdbs | wc -l


