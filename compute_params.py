#!/usr/bin/env python
# -*- coding: utf-8 -*-

#python main_profiles.py /home/mzieleniewska/empi/from_hpc/data/smp/

from __future__ import division

import argparse

import os
import glob
import json
import pandas as pd
import numpy as np
from scipy.stats import scoreatpercentile, gamma, skew, pareto, entropy
import matplotlib.pyplot as plt
import nolds
from scipy.stats.stats import pearsonr

from create_histograms import _get_histogram_values_in_sec
from mark_spindles_mmp import fit_gaussian_mixture_model, compute_autocorrelation


pages_db = "./pages_db.json"
fs = 128.


def get_total_number_of_epochs(f_name):
	with open(pages_db) as f:
		pages_data = json.load(f)
		try:
			number_of_epochs = pages_data[f_name]
		except IOError:
			"No key in json file!"
	return number_of_epochs


def plot_gamma(x_diff, new_bins, fit_alpha, fit_loc, fit_beta, skewness):
	fig = plt.figure(figsize=(20, 15))
	ax = fig.add_subplot(111)
	n = ax.hist(x_diff, 100, normed=True, histtype='stepfilled', alpha=0.2)
	bins = n[1]
	new_bins = []
	for b in xrange(len(bins)-1):
		new_bins.append((bins[b]+bins[b+1])*0.5)
	new_bins = np.array(new_bins)
	# ax.plot(new_bins, pareto.pdf(new_bins, b=fit_alpha, loc=fit_loc, scale=fit_beta), 'r-', linewidth=4) #for pareto
	ax.plot(new_bins, gamma.pdf(new_bins, a=fit_alpha, loc=fit_loc, scale=fit_beta), 'r-', linewidth=4) #for gamma
	ax.annotate('shape = %.8f\nloc = %.8f\nscale = %.8f, \nskew = %.8f' %(fit_alpha, fit_loc, fit_beta, skewness), \
				xy=(new_bins[len(new_bins)/2],n[0][1]), ha='left', fontsize=20)
	return fig


def plot_aurocorrelation(df_hist, time, corr, fit_fn, dt):
	fig = plt.figure(figsize=(40, 22))
	fig.add_subplot(311)
	plt.bar(df_hist.time, df_hist.occurences, width=dt)	
	fig.add_subplot(312)
	plt.title('Autocorreation, var_corr: ' + str(var_corr) + ' mean: ' + str(np.mean(np.abs(corr))), fontsize=20)
	plt.plot(time, corr, 'y', time, fit_fn(time), '--k')
	fig.add_subplot(313)
	plt.title('Autocorreation, var_corr: ' + str(np.var(fit_fn(time))) + ' mean: ' + str(np.mean(np.abs(fit_fn(time)))), fontsize=20)
	plt.plot(time, corr-fit_fn(time), 'm')
	return fig


def find_nearest(array, value):
	return (np.abs(array - value)).argmin()


def compute_correlation_of_profiles(directory, name, number_of_epochs):
	# computes correlation between two profiles

	name_pattern = "{}/{}".format(directory, name) + "_{}_occ.csv"

	total_len = int(number_of_epochs * 20) #in sec

	df_ss = pd.read_csv(name_pattern.format("spindle"), index_col=0)
	df_swa = pd.read_csv(name_pattern.format("SWA"), index_col=0)

	df_ss = df_ss.drop(df_ss[df_ss.amplitude < 20.].index, inplace=False).reset_index(drop=True)
	df_swa = df_swa.drop(df_swa[df_swa.amplitude < 75.].index, inplace=False).reset_index(drop=True)

	hist_ss = _get_histogram_values_in_sec(df_ss, total_len, 20, profile_type='energy') 
	hist_swa = _get_histogram_values_in_sec(df_swa, total_len, 20, profile_type='energy') 

	return pearsonr(hist_ss.occurences, hist_swa.occurences)


def compute_params(df, number_of_epochs, structure):

	total_len = int(number_of_epochs * 20) #in sec

	if structure=='spindle':
		profile_type = 'energy'
		bin_width = 20
	elif structure=='SWA':
		profile_type = 'energy'
		bin_width = 20

	df_hist = _get_histogram_values_in_sec(df, total_len, bin_width, profile_type=profile_type) #in sec
	occ = df_hist.occurences
	time = df_hist.time
	dt = time[1]-time[0]
	
	if (structure=='SWA') and (profile_type=='percent'):
		occ[occ < 20.] = 0

	#power = np.sum(df['amplitude']**2) / total_len
	power = np.sum(df['modulus']) / total_len

	profile_dfa = nolds.dfa(occ)

	if len(df) > 2:
		if structure == 'spindle':
			frequency_mse = np.sum((df['frequency'] - 13) ** 2) / (len(df['frequency']) - 1)
			frequency_var = np.var(df['frequency'], ddof=1)
			df_params = pd.DataFrame([[power, frequency_mse, frequency_var, \
									   profile_dfa]], 
									 columns=['power_spindle', 'frequency_mse_spindle', 'frequency_var', \
									 		  'profile_dfa_spindle'])
		elif structure == 'SWA':
			df_hist_deep = _get_histogram_values_in_sec(df, total_len, 20, profile_type='percent')
			occ_deep_sleep = df_hist_deep.occurences
			min_deep_sleep = len(occ_deep_sleep[occ_deep_sleep >= 20.]) * 20. / 60. #in min
			min_deep_sleep_50 = len(occ_deep_sleep[occ_deep_sleep >= 50.]) * 20. / 60. #in min
			occ_deep_sleep[occ_deep_sleep < 20.] = 0.
			dfa_deep_sleep = nolds.dfa(occ_deep_sleep)
			deep_sleep_ratio = min_deep_sleep ** dfa_deep_sleep
			occ_deep_sleep[occ_deep_sleep < 50.] = 0.
			dfa_deep_sleep_50 = nolds.dfa(occ_deep_sleep)
			deep_sleep_ratio_50 = min_deep_sleep_50 ** dfa_deep_sleep_50
			df_params = pd.DataFrame([[power, profile_dfa, min_deep_sleep, \
									   dfa_deep_sleep, deep_sleep_ratio, min_deep_sleep_50, \
									   dfa_deep_sleep_50, deep_sleep_ratio_50]], 
									 columns=['power_SWA', 'profile_dfa_SWA', 'min_deep_sleep', \
									 		  'dfa_deep_sleep', 'deep_sleep_ratio', 'min_deep_sleep_50', \
									 		  'dfa_deep_sleep_50', 'deep_sleep_ratio_50'])
	else:
		if structure == 'spindle':
			df_params = pd.DataFrame([[0, np.inf, np.inf, \
									   0.5]], 
									 columns=['power_spindle', 'frequency_mse_spindle', 'frequency_var', \
									 		  'profile_dfa_spindle'])
		elif structure == 'SWA':
			df_params = pd.DataFrame([[0, 0.5, 0, \
									   0.5, 0, 0, \
									   0.5, 0]], 
									 columns=['power_SWA', 'profile_dfa_SWA', 'min_deep_sleep', \
									 		  'dfa_deep_sleep', 'deep_sleep_ratio', 'min_deep_sleep_50', \
									 		  'dfa_deep_sleep_50', 'deep_sleep_ratio_50'])

	return df_params


def get_spectrum_parameters(spectrum_dir, f_name):
	bands = {'delta': [0.5, 2], 'theta': [2, 8], 'alpha': [8, 12] ,'beta1': [12, 16], 'beta2': [16, 25]}
	df_temp = pd.DataFrame(columns=['spectral_entropy'])
	spectrum = np.load(os.path.join(spectrum_dir, f_name + '_ears_full_128_power.npy'))
	frequencies = np.load(os.path.join(spectrum_dir, f_name + '_ears_full_128_freq.npy'))
	for b in bands.keys():
		band = bands[b]
		p_rel = np.mean(spectrum[np.where((frequencies >= band[0]) & (frequencies < band[1]))[0]]) / np.mean(spectrum)
		p = np.mean(spectrum[np.where((frequencies >= band[0]) & (frequencies < band[1]))[0]])
		df_temp[b + "_rel_power"] = [p_rel]
		# df_temp[b + "_power"] = [p]
	se = 0
	for idf in xrange(len(frequencies)):
		se += - (np.sum(spectrum[idf] * np.log2(spectrum[idf]))) 
	se = se / np.log2(idf+1)
	df_temp["spectral_entropy"] = [se]
	return df_temp


def get_clustering_parameters(clustering_file, f_name, structure):
	df = pd.read_csv(clustering_file)
	ds = df[df["rec_id"].str.contains(f_name)]
	ds = ds.drop(["rec_id", "calinski_harabaz_p", "F0_p", "F0"], axis=1, inplace=False).reset_index(drop=True)
	ds = ds.rename(index=str, columns={"calinski_harabaz": "calinski_harabaz_"+structure, "nb_clustering_labels": "nb_clustering_labels_"+structure})
	return ds


def main():
	parser = argparse.ArgumentParser(description='Compute classification parameters.')
	parser.add_argument('files', nargs='+', metavar='file', help='path to *.b files')
	namespace = parser.parse_args()

	structures = ['spindle','SWA']
	out_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader' #patients lub control_99rms_new_reader
	spectrum_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/patients_spectra_Cz_05_25Hz/'

	add_spectrum = True  #False for control

	for path_to_b in namespace.files:
		full_name = os.path.basename(path_to_b).split('.')[0]
		# name = full_name  #uncomment for control
		name = "_".join(full_name.split("_")[:4])
		print name
		number_of_epochs = get_total_number_of_epochs(name)
		df_params = pd.DataFrame()
	 	for structure in structures:
	 	 	df = pd.read_csv(os.path.join(out_dir, 'occ_results', name + '_' + structure + '_occ_sel.csv'), index_col=0)
	 	 	df_temp = compute_params(df, number_of_epochs, structure)
 		 	if structure=='spindle':
	 	 		clustering_file = os.path.join(out_dir, "clustering_parameters_spindle_20uV.csv")
	 	 	elif structure=='SWA':
	 	 		clustering_file = os.path.join(out_dir, "clustering_parameters_SWA_75uV.csv")
	 		df_cluster = get_clustering_parameters(clustering_file, name, structure)
	 		# comb = pd.DataFrame([{'ch_dfa_' + structure: float(df_cluster['calinski_harabaz_' + structure]) ** float(df_temp['profile_dfa_' + structure])}])			
	 		df_params = pd.concat([df_params, df_temp, df_cluster.reset_index(drop=True)], axis=1)
	 	if add_spectrum:
			df_spectrum = get_spectrum_parameters(spectrum_dir, name)
			df_params = pd.concat([df_params, df_spectrum], axis=1)
		if not os.path.exists(os.path.join(out_dir, 'params')):
			os.makedirs(os.path.join(out_dir, 'params'))
		df_params.to_csv(os.path.join(out_dir, 'params', name+'_params.csv'))



if __name__ == '__main__':
	main()

	# group = 'patients'
	# mp_type = 'smp'
	# out_dir = 'patients_99rms_new_reader'

	# book_dir = os.path.join('/home/mzieleniewska/empi/from_hpc/data/', mp_type)
	# spectrum_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/patients_spectra_Cz_05_25Hz/'
	# clustering_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/'

	# data_dir = os.path.join(book_dir, out_dir)

	# structures = ['spindle','SWA']#['alpha', 'theta', 'beta']

	# file_list = glob.glob(os.path.join(book_dir, group+'_decomposed_books', "*.b"))

	# for id_f in xrange(0, len(file_list)):
	# 	f = file_list[id_f]
	# 	if group=='patients':
	# 		f_parts = os.path.split(f)[1].split('.')[0].split('_')
	# 		name = '_'.join(f_parts[:4])
	# 		extension = 'ears_full_128_'+mp_type
	# 	else:
	# 		name = os.path.split(f)[1].split('_')[0]+'_ears'
	# 		extension = mp_type
	# 	number_of_epochs = get_total_number_of_epochs(name)

	# 	df_params = pd.DataFrame()

	# 	directory = os.path.join(book_dir, out_dir, 'occ_results')

	# 	print name

	# 	for structure in structures:
	# 		df = pd.read_csv(os.path.join(directory, name+'_'+structure+'_occ_sel.csv'), index_col=0)

	# 		df_temp = compute_params(df, number_of_epochs, structure)
	# 		if structure=='spindle':
	# 			clustering_file = os.path.join(clustering_dir, "clustering_parameters_spindle_20uV.csv")
	# 		elif structure=='SWA':
	# 			clustering_file = os.path.join(clustering_dir, "clustering_parameters_SWA_75uV.csv")

	# 		df_cluster = get_clustering_parameters(clustering_file, name, structure)
	# 		# comb = pd.DataFrame([{'ch_dfa_' + structure: float(df_cluster['calinski_harabaz_' + structure]) ** float(df_temp['profile_dfa_' + structure])}])			
	# 		df_params = pd.concat([df_params, df_temp, df_cluster.reset_index(drop=True)], axis=1)
	# 	df_spectrum = get_spectrum_parameters(spectrum_dir, name)
	# 	df_params = pd.concat([df_params, df_spectrum], axis=1)

	# 	if not os.path.exists(os.path.join(book_dir, out_dir, 'params')):
	# 		os.makedirs(os.path.join(book_dir, out_dir, 'params'))
	# 	df_params.to_csv(os.path.join(book_dir, out_dir, 'params', name+'_params.csv'))
