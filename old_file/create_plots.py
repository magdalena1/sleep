#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

WAVE_TYPES = ['SS','SWA']
ylabels = {'SS':'num. of spindles/3min', 'SWA':'%SWA/20sec'}#, 'Amp/3min', 'Amp/3min', 'Amp/3min', 'Amp/3min'}


def create_plots(name, directory):
	data_name = "{}/{}".format(directory, name) + "_{}_hipnogram.csv"
	fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		hipnogram = pd.read_csv(data_name.format(wave_type))
		occ = hipnogram['occurences']
		names_occ = hipnogram['time']
		bar_width = names_occ[2]-names_occ[1]
		axs[i].bar(names_occ, occ, width=bar_width, color='#c8c6d1', edgecolor='#626166')
		axs[i].set_xlim([names_occ.iloc[0], names_occ.iloc[-1]])
		axs[i].set_ylabel(ylabels[wave_type])
		if i==(len(WAVE_TYPES)-1): axs[i].set_xlabel('time [hours]')
	return fig

if __name__ == '__main__':
	# parser = argparse.ArgumentParser('Create histrograms for profiles')
	# parser.add_argument('--name', required=True,
	# 				help='Base name of profiles (without extension, e.g. *_<wave>_hipnogram.csv).')
	# parser.add_argument('--dir', required=True, help='Path to directory')
	# args = parser.parse_args()

	name = 'WS_31_07_2016_C3-ear_128'

	dir = '/home/mzieleniewska/sleep2/books/one_channel/WS_31_07_2016_C3-ear_128/'

	fig = create_plots(name, dir)

	fig.suptitle('runrun', fontsize=16)

	plt.show()

	# create_plots(args.name, args.dir)

