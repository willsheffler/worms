#!/bin/bash
#2019-01-27
#This is the script to launch from the command line. It will iterate through a list of config files, launching the worms cmd.sh script.


while read line; do
	echo ${line}
	./cmd.sh ${line}
done < config/config.list

