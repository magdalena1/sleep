#!/bin/bash

# usage: 
# ./plot_profiles_single_channel.sh /home/mzieleniewska/sleep2/books/one_channel

DIR=$1

for FILENAME in $DIR/*.b
do
	WORKDIR="$FILENAME"
	# echo "${FILENAME%%.*}"
	BASENAME=$(basename "$FILENAME")
	BASENAME="${BASENAME%.*}"
	
	#python create_profiles_single_channel.py --name $FILENAME
	python save_structures_params.py --name $BASENAME --dir $DIR/$BASENAME
 	#python create_histograms.py --name $BASENAME --dir $DIR/$BASENAME --bin-width 3
	#Rscript --vanilla create_plots.R $DIR/$BASENAME/$BASENAME
done


