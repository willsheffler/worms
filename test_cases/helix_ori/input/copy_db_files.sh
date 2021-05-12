# dbfile='input/local_HFuse_het_2chain_2arm_database.Sh13-5+1-AI_2.20180905.txt'
dbfile='/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180516.txt'
for i in $(grep file $dbfile | sed -e s=\{\"file\"\:\ \"==g | sed -e s=\",==g); do
   rsync -avz fw.bakerlab.org:$i ./dbfile_pdbs; 
done
grep file $dbfile | wc -l
ls dbfile_pdbs | wc -l


