# this is an evil script for copying baker worms database files to a local machine...
# probably better to copy your worms cache instead

for i in /home/rubul/database/fusion/hb_dhr/master_database_generation2.txt \
/home/yhsia/helixdock/database/HBRP_Cx_database.txt \
/home/rubul/database/fusion/hb_dhr/c6_database.txt \
/home/yhsia/helixfuse/rosetta_scripts_ver/processing/database/HFuse_Cx_database.20180219.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.txt \
/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180416.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180416.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180416.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180416.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180416.txt \
/home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180416.txt \
  ; do
    echo $i
    mkdir -p $(dirname $i);
    rsync -a -e 'ssh' digs:$i $i;
    for j in $(grep file $i | sed -e s=.*\"/home=/home=g -e s=.*\"/work=/work=g  -e s=\",==g); do
        if [ ! -f $j ]; then
            echo '   ' $j
            mkdir -p $(dirname $j);
            rsync -a -e 'ssh' digs:$j $j;
        fi
    done

done


# /home/baker/will_fusion/scripts/generate_chains.py --config_file PS1_C3_CN_CNNC.config --max_samples 1.e11 --max_chunk_length 180 --nres_from_termini 80 --min_chunk_length 85 --use_class True --prefix PS1_C3_CN_CNNC_n22 --err_cutoff 1.5 --num_contact_threshold 30 --max_chain_length 400 --min_seg_length 15 --cap_number_of_pdbs_per_segment 500 --clash_cutoff 10.0 --superimpose_rmsd 1.0 --superimpose_length 9 --output_resfile_info_in_pdb True --Nproc_for_sympose 2 --max_number_of_fusions_to_evaluate 100000 --database_files /home/rubul/database/fusion/hb_dhr/master_database_generation2.txt /home/yhsia/helixdock/database/HBRP_Cx_database.txt /home/rubul/database/fusion/hb_dhr/c6_database.txt /home/yhsia/helixfuse/rosetta_scripts_ver/processing/database/HFuse_Cx_database.20180219.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.txt /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180416.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180416.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180416.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180416.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180416.txt /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180416.txt