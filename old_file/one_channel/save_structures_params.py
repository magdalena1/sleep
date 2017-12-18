#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pylab as py
import pandas as pd
import numpy as np

import os

import IPython

WAVE_TYPES = ['SS', 'SWA', 'delta', 'theta', 'alpha', 'beta'] 

fs = 128.0

def save_structures_params(name, directory):
	data_name = os.path.join(directory, name + "_{}.csv")
	data_out_name = os.path.join(directory, name + "_params.csv")
	params = {}
	for (i, wave_type) in enumerate(WAVE_TYPES):
		data = pd.read_csv(data_name.format(wave_type))
		n = len(data)
		params[wave_type+'_freq'] = np.mean(data['frequency'])
		params[wave_type+'_freq_std'] = np.std(data['frequency'])
		params[wave_type+'_width'] = np.mean(data['width'])
		params[wave_type+'_width_std'] = np.std(data['width'])
		params[wave_type+'_amp'] = np.mean(data['amplitude'])
		params[wave_type+'_amp_std'] = np.std(data['amplitude'])
	df = pd.DataFrame(data=params, index=[0])
	df.to_csv(data_out_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Create histograms for profiles')
	parser.add_argument('--name', required=True,
					help='Base name of profiles (without suffix eg. "_alpha.csv").')
	parser.add_argument('--dir', required=True, help='Path to directory')
	args = parser.parse_args()

	save_structures_params(args.name, args.dir)
	# name = '/Users/magdalena/projects/python/sleep_decompositions/part3/AC_C4-RM_128.b' #args.name
	# directory = '/Users/magdalena/projects/python/sleep_decompositions/part3/'



