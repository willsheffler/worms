#!/bin/bash
#2019-10-10
#DO NOT RUN THIS SCRIPT FROM THE COMMAND LINE
#This is the command shell script that will be run by the meta_command shell script.
#You may need to edit the memory and/or queue

#Locate config file
config="config/${1}"
config_name=`echo ${config##*/} |cut -d'.' -f1`

#Locate flags file. This file should be edited per job, as needed.
flags="worms.flags"

if [[ ! -e output ]]; then mkdir output/; fi
if [[ ! -e output/${config_name} ]]; then mkdir output/${config_name}/; fi
outpath="output/${config_name}"


echo "OMP_NUM_THREADS=1 PYTHONPATH=/home/sheffler/src/worms_beta /home/sheffler/.conda/envs/worms/bin/python -mworms @${flags} --config ${config} --output_prefix ${outpath}/${config_name}"
exit

#Instructions for SLURM
sbatch \
-p medium \
-N 1 \
-n 8 \
-J worms_${config_name} \
--output ${outpath}/worms_${config_name}_job.log \
--err ${outpath}/worms_${config_name}_job.err \
--mem 30G \
--wrap="OMP_NUM_THREADS=1 PYTHONPATH=/home/sheffler/src/worms_beta /home/sheffler/.conda/envs/worms/bin/python -mworms \
@${flags} \
--config ${config} \
--output_prefix ${outpath}/${config_name}"

