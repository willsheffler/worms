#!/bin/bash
#2019-10-14

#This script will create a worms flag file using the databases in the input folder

if [[ -e worms.flags ]]; then rm worms.flags; fi

#Copy and rename template flag file
cp input/template.flags ./worms.flags

for file in ./input/*.txt; do
	path=`readlink -f ${file}`
	echo -ne "\t${path}\n" >> ./worms.flags
done
	
