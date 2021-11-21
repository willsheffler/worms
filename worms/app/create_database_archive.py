import sys, os
import worms

def main():
   for dbfile in sys.argv[1:]:
      dbname = os.path.basename(dbfile)
      dbname = dbname.replace('.txt', '').replace('.json', '').replace('.', '_')
      newfile = dbname + '.txz'
      print('making local copy of', dbfile)
      print('    ', dbname)
      print('    ', newfile)
      worms.database.archive.make_bblock_archive(
         dbfiles=[dbfile],
         target=newfile,
         dbname=dbname,
         overwrite=False,
      )

if __name__ == '__main__':
   main()
