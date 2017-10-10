#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import glob, os
import json
from scipy.signal import argrelextrema

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from collections import defaultdict

from create_plots import create_plots

WAVE_TYPES = ['SS','SWA']

pages_db = '/home/mzieleniewska/sleep2/eeg_profiles/one_channel/pages_db_single_channel.json'

if __name__ == '__main__':

	out_dir = '/home/mzieleniewska/sleep2/results/distributions'
	book_dir = '/home/mzieleniewska/sleep2/books/one_channel'

	file_list = glob.glob(os.path.join(book_dir, "*.b"))

	for f in [file_list[11]]:

		patient = os.path.basename(f).split('.')[0]
		print "Working on patient: ", patient
		patient_dir = os.path.join(book_dir, patient)
		data_name = "{}/{}".format(patient_dir, patient) + "_{}.csv"

		with open(pages_db) as pages_f:
			pages_data = json.load(pages_f)
			total_len = float(pages_data[patient]*20)

		fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))

		densities = defaultdict(list)

		for i, wave_type in enumerate(WAVE_TYPES):

			data = pd.read_csv(data_name.format(wave_type))
			data['absolute_position'] = ((data['book_number']-1)*20. + data['position'])

			X = data['absolute_position'][:, np.newaxis]
			time = np.linspace(0, total_len, 3*total_len+1)[:, np.newaxis]

			grid = GridSearchCV(KernelDensity(),
								{'bandwidth': np.linspace(20, 3600, 20)},
								cv=50) # 20-fold cross-validation
			grid.fit(X)
			best_band = np.round(grid.best_params_['bandwidth'])
			
			# fig = create_plots(patient, patient_dir)
			binBoundaries = np.linspace(0, total_len, total_len/180+1)
			axs[i].hist(X[:,0], bins=binBoundaries, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)

			kde = KernelDensity(kernel='gaussian', bandwidth=900).fit(X)
			log_dens = kde.score_samples(time)
			dens = np.exp(log_dens)
			densities[wave_type] = dens
			axs[i].plot(time[:, 0], np.exp(log_dens), '-', label='bw=%.2f' % kde.bandwidth)
			axs[i].legend(loc='upper left')

			mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
			print "nb. of maxima: ", len(ma)
			print "optimal bandwidth: ", best_band

			diffs = np.diff(data['absolute_position'].sort_values())

			autocorr = np.correlate(dens, dens, mode='full')
			autocorr = autocorr[int(autocorr.size/2):]


			# ### Poisson
			# mu = np.mean(diffs)
			# mean, var, skew, kurt = st.poisson.stats(mu, moments='mvsk')

			# D, p_value = st.kstest(diffs/diffs.max(), 'uniform') #Kolmogorov-Smirov test


			# ### ???
			# fig, ax = plt.subplots(1, 1)
			# x = np.arange(st.poisson.ppf(0.01, mu),
			# 			  st.poisson.ppf(0.99, mu))
			# ax.plot(x, st.poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
			# ax.vlines(x, 0, st.poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
			# prob = st.poisson.cdf(x, mu)

			# my_pdf = st.kde.gaussian_kde(X[:,0])
			# my_cdf = lambda ary: np.array([my_pdf.integrate_box_1d(-np.inf, x) for x in ary])
			# D2, p_value2 = st.kstest(X[:,0], my_cdf)
			# ###
			
		plt.show()


		# fig.suptitle('p-value = '+str(p_value), fontsize=16)
		# plt.savefig(os.path.join(out_dir, patient+'.png'), bbox_inches='tight')
		# plt.close(fig)


		#  ks_2samp