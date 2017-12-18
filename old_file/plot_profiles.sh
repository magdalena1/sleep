#!/bin/bash

# usage: 
# ./plot_profiles_single_channel.sh /Users/magdalena/projects/python/sleep_decompositions/part3

DIR=$1
CHANNEL=$2

for FILENAME in $DIR/*.b
do
	WORKDIR="$FILENAME"
	# echo "${FILENAME%%.*}"
	BASENAME=$(basename "$FILENAME")
	BASENAME="${BASENAME%.*}"

	#echo {$BASENAME\_$CHANNEL}
	
	#python create_profiles_single_channel.py --name $FILENAME
	#python save_structures_params.py --name $BASENAME --dir $DIR/$BASENAME
 	
 	#python create_histograms.py --name $BASENAME --channel $CHANNEL --dir $DIR/$BASENAME --bin-width 3
	Rscript --vanilla create_plots.R $DIR/$BASENAME/$BASENAME\_$CHANNEL
done


