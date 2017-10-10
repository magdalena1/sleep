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
import numpy as np

import glob
import re

from obci.analysis.obci_signal_processing.tags import tags_file_writer as tags_writer
from obci.analysis.obci_signal_processing.tags import tag_utils

channels_names = ['Fp1','Fp2','AFz',
				  'F7','F3','Fz','F4','F8',
				  'T3','C3','Cz','C4','T4',
				  'T5','P3','Pz','P4','T6',
				  'O1','O2']  

structures = ['SS', 'SWA']#,'KC','alpha','theta']				

def svarog_tags_writer(f_name, dir_name, fs):
	out_file = os.path.join(dir_name, f_name + '.tag')
	writer = tags_writer.TagsFileWriter(out_file)
	for structure in structures:
		files = glob.glob(os.path.join(dir_name, f_name, f_name + '_*_' + structure + '.csv'))
		for csv_file in files:
			search_pattern = os.path.join(dir_name, f_name, f_name + '_(.*)_' + structure + '.csv')
			result = re.search(search_pattern, csv_file)
			ch = result.group(1)
			idx = np.where(np.array(channels_names)==ch)[0][0]
			df = pandas.read_csv(csv_file)
			offset = df['offset']
			width = df['width']
			amp = df['amplitude']
			for i in xrange(len(offset)):
				tag = {'channelNumber':idx, 'start_timestamp':offset[i]/fs, 'end_timestamp':offset[i]/fs+width[i], 
					   'name':structure, 'desc':{'amplitude':amp[i], 'width':width[i]}}
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

