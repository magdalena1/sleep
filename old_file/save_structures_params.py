#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pylab as py
import pandas as pd
import numpy as np
import json
import os


WAVE_TYPES = ['SS', 'delta', 'theta', 'alpha', 'beta'] 

channels_db = "./channels_db.json"

fs = 128.0

def save_structures_params(name, directory):
	data_name = "{}/{}_{}.csv"
	data_out_name = os.path.join(directory, name + "_params.csv")
	params = {}
	for (i, wave_type) in enumerate(WAVE_TYPES):
		patient_id = name.split('_')[0]
		ids = patient_id + '_' + wave_type
		with open(channels_db) as f:
			channels_data = json.load(f)
			channel = channels_data[ids]
		data = pd.read_csv(data_name.format(directory, name + '_' + channel, wave_type))
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




