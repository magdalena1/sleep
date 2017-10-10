#!/usr/bin/env python
# -*-coding: utf-8-*-

"""
Function for generating tags for selected structures:
python tags_writer.py --name 'WS_31_07_2016_longitudinal_full_128' --dir . --fs 128

"""

import argparse
import sys
import pandas
import os.path

from obci.analysis.obci_signal_processing.tags import tags_file_writer as tags_writer
from obci.analysis.obci_signal_processing.tags import tag_utils

structures = ['SS', 'SWA']#,'KC','alpha','theta']				

def svarog_tags_writer(f_name, dir_name, fs):
	out_file = os.path.join(dir_name, f_name + '.tag')
	writer = tags_writer.TagsFileWriter(out_file)
	for structure in structures:
		csv_file = os.path.join(dir_name, f_name + '_' + structure + '.csv')
		df = pandas.read_csv(csv_file)
		offset = df['offset']
		width = df['width']
		amp = df['amplitude']
		for i in xrange(len(offset)):
			tag = tag_utils.pack_tag_to_dict(offset[i]/fs, offset[i]/fs+width[i], 
											 structure,{'amplitude':amp[i], 'width':width[i]}, u'10')
			writer.tag_received(tag)
	writer.finish_saving(0.0)

def main():
	parser = argparse.ArgumentParser('Create tags for structures')
	parser.add_argument('--name', required=True, help='Base filename (without suffix e.g. "_alpha.csv")')
	parser.add_argument('--dir', required=True, help='Path to directory')
	parser.add_argument('--fs', required=True, help='Sampling frequency')
	args = parser.parse_args()
	svarog_tags_writer(args.name, args.dir, float(args.fs))

if __name__ == "__main__":
	main()

