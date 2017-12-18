#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pylab as py
import pandas as pd
import numpy as np
import json
import IPython
import os

WAVE_TYPES = ['theta']#['spindle', 'SWA']#, 'delta', 'theta', 'alpha', 'beta'] 
PROFILE_TYPES = ['counts', 'percent']
ylabels = {'spindle':'num. of spindles/3min', 'SWA':'%SWA/20sec'}#, 'SWA':'%SWA/20sec', 'Amp/3min', 'Amp/3min', 'Amp/3min', 'Amp/3min'}

pages_db = './pages_db.json'

channels_db = "./channels_db.json"

fs = 128.0

def create_plots(name, directory, annotate=True):
	data_name = "{}/{}".format(directory, name) + "_{}_hipnogram.csv"

	if annotate:
		global_dir = "/".join(directory.split("/")[:-1])
		params_csv = os.path.join(global_dir, 'params', name + "_params.csv")
		params = pd.read_csv(params_csv)
		params.drop([u'Unnamed: 0', u'frequency_mse_spindle', u'spectral_entropy', u'theta_rel_power', \
					 u'alpha_rel_power', u'beta2_rel_power', u'beta1_rel_power', u'delta_rel_power'], axis=1, inplace=True)

	fig, axs = py.subplots(len(WAVE_TYPES), 1, figsize=(11.69, 8.27))
	# d = params.filter(regex=wave_type)
	text = []
	for c in params.columns:
		text.append(c + ' = %.4f' % (params[c].tolist()[0]))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		hipnogram = pd.read_csv(data_name.format(wave_type))
		occ = hipnogram['occurences']
		names_occ = hipnogram['time']
		bar_width = names_occ[2]-names_occ[1]
		axs[i].bar(names_occ, occ, width=bar_width, color='#c8c6d1', edgecolor='#626166')
		axs[i].set_xlim([names_occ.iloc[0], names_occ.iloc[-1]])
		axs[i].set_ylabel(ylabels[wave_type])
		if i==(len(WAVE_TYPES)-1): axs[i].set_xlabel('time [hours]')
	axs[i].annotate('\n'.join(text), xy=(0.3, 1), xytext=(0.7, 0.8), xycoords='figure fraction', fontsize=12)

	return fig


def create_amplitude_plots(name, directory, annotate=True):
	data_name = "{}/{}".format(directory, name) + "_{}_occ.csv"

	with open(pages_db) as f:
		key = name.split('.')[0]
		pages_data = json.load(f)
		total_len = int(pages_data[key] * 20) #in sec

	if annotate:
		global_dir = "/".join(directory.split("/")[:-1])
		params_csv = os.path.join(global_dir, 'params', name + "_params.csv")
		params = pd.read_csv(params_csv)
		params.drop([u'Unnamed: 0', u'frequency_mse_spindle', u'spectral_entropy', u'theta_rel_power', \
					 u'alpha_rel_power', u'beta2_rel_power', u'beta1_rel_power', u'delta_rel_power'], axis=1, inplace=True)

	fig, axs = py.subplots(len(WAVE_TYPES), 1, figsize=(11.69, 8.27))
	# d = params.filter(regex=wave_type)
	text = []
	for c in params.columns:
		text.append(c + ' = %.4f' % (params[c].tolist()[0]))

	for (i, wave_type) in enumerate(WAVE_TYPES):
		t = np.linspace(0, total_len, total_len * 128.) #in samples
		x = np.zeros(len(t))
		occ = pd.read_csv(data_name.format(wave_type))
		for index, row in occ.iterrows():
			idx = np.abs(t -row["absolute_position"]).argmin()
			x[idx] += row["amplitude"]
		axs[i].plot(t, x)
		if i==(len(WAVE_TYPES)-1): axs[i].set_xlabel('time [sec]')
	axs[i].annotate('\n'.join(text), xy=(0.3, 1), xytext=(0.7, 0.8), xycoords='figure fraction', fontsize=12)

	return fig


# def create_combo_plot(name, directory, annotate=True):

# 	data_hipnogram = "{}/{}".format(directory, name) + "_{}_hipnogram.csv"
# 	data_occ = "{}/{}".format(directory, name) + "_{}_occ_sel.csv"

# 	with open(pages_db) as f:
# 		key = name.split('.')[0]
# 		pages_data = json.load(f)
# 		total_len = int(pages_data[key] * 20) #in sec

# 	if annotate:
# 		global_dir = "/".join(directory.split("/")[:-1])
# 		params_csv = os.path.join(global_dir, 'params', name + "_params.csv")
# 		params = pd.read_csv(params_csv)
# 		params.drop([u'Unnamed: 0', u'frequency_mse_spindle', u'spectral_entropy', u'theta_rel_power', \
# 					 u'alpha_rel_power', u'beta2_rel_power', u'beta1_rel_power', u'delta_rel_power'], axis=1, inplace=True)

# 	fig, axs = py.subplots(2 * len(WAVE_TYPES), 1, figsize=(11.69, 8.27))
# 	text = []
# 	for c in params.columns:
# 		text.append(c + ' = %.4f' % (params[c].tolist()[0]))

# 	t = np.linspace(0, total_len, total_len) #in sec * 128.) #in samples
# 	s_i = 0

# 	for (i, wave_type) in enumerate(WAVE_TYPES):
# 		hipnogram = pd.read_csv(data_hipnogram.format(wave_type))
# 		counts = hipnogram['occurences']
# 		names_occ = hipnogram['time']
# 		bar_width = names_occ[2]-names_occ[1]
# 		axs[s_i].bar(names_occ, counts, width=bar_width, color='#c8c6d1', edgecolor='#626166')
# 		axs[s_i].set_xlim([names_occ.iloc[0], names_occ.iloc[-1]])

# 		if wave_type == "spindle":
# 			axs[s_i].set_ylim([0, 30])
# 		else:
# 			axs[s_i].set_ylim([0, 100])
# 			axs[s_i].axhline(y=20., color='k', linestyle='-')
# 			axs[s_i].axhline(y=50., color='k', linestyle='-')

# 		axs[s_i].set_ylabel(ylabels[wave_type])
# 		if i==(len(WAVE_TYPES)-1): axs[i].set_xlabel('time [hours]')
# 		s_i += 1
# 		x = np.zeros(len(t))
# 		occ = pd.read_csv(data_occ.format(wave_type))
# 		for index, row in occ.iterrows():
# 			idx = np.abs(t -row["absolute_position"]).argmin()
# 			x[idx] += row["amplitude"]
# 		axs[s_i].plot(t, x)
# 		axs[s_i].set_xlim([t[0], t[-1]])

# 		if wave_type == "spindle":
# 			axs[s_i].set_ylim([0, 172])
# 		else:
# 			axs[s_i].set_ylim([0, 800])

# 		if wave_type == "spindle":
# 			ss_count = max(counts)
# 			ss_amp = max(x)
# 		else:
# 			swa_amp = max(counts)
# 		s_i += 1

# 	axs[0].annotate('\n'.join(text), xy=(0.3, 1), xytext=(0.7, 0.8), xycoords='figure fraction', fontsize=12)

# 	return fig, ss_count, ss_amp, swa_amp


def profile_energy_plot(name, directory, additional_parameter='energy', annotate=True):

	#data_occ = "{}/{}".format(directory, name) + "_{}_occ_sel.csv"
	data_occ = "{}/{}".format(directory, name) + "_{}_occ.csv"

	with open(pages_db) as f:
		key = name.split('.')[0]
		pages_data = json.load(f)
		total_len = int(pages_data[key] * 20) #in sec

	if annotate:
		global_dir = "/".join(directory.split("/")[:-1])
		params_csv = os.path.join(global_dir, 'params', name + "_params.csv")
		params = pd.read_csv(params_csv)
		params.drop([u'Unnamed: 0', u'frequency_mse_spindle', u'spectral_entropy', u'theta_rel_power', \
					 u'alpha_rel_power', u'beta2_rel_power', u'beta1_rel_power', u'delta_rel_power'], axis=1, inplace=True)
		text = []
		for c in params.columns:
			text.append(c + ' = %.4f' % (params[c].tolist()[0]))

	fig, axs = py.subplots(2 * len(WAVE_TYPES), 1, figsize=(11.69, 8.27))

	s_i = 0

	for (i, wave_type) in enumerate(WAVE_TYPES):
		data = pd.read_csv(data_occ.format(wave_type))
		if wave_type=='spindle':
			histogram = _get_histogram_values_in_sec(data, total_len, 20, 'counts')
		elif wave_type == 'SWA':
			histogram = _get_histogram_values_in_sec(data, total_len, 20, 'percent')
		else:
			histogram = _get_histogram_values_in_sec(data, total_len, 20, 'counts')
		counts = histogram['occurences']
		names_occ = histogram['time'] / 3600
		bar_width = names_occ[2] - names_occ[1]
		axs[s_i].bar(names_occ, counts, width=bar_width, color='#c8c6d1', edgecolor='#626166')
		axs[s_i].set_xlim([names_occ.iloc[0], names_occ.iloc[-1]])
		if wave_type == "spindle":
			axs[s_i].set_ylim([0, 15])
			axs[s_i].set_ylabel('num. of sleep\nspindles / 20 s')
		elif wave_type == 'SWA':
			axs[s_i].set_ylim([0, 100])
			axs[s_i].axhline(y=20., color='k', linestyle='-', linewidth=0.5)
			axs[s_i].axhline(y=50., color='k', linestyle='-', linewidth=0.5)
			axs[s_i].set_ylabel('%SWA / 20 s')
		else:
			axs[s_i].set_ylim([0, 15])
			axs[s_i].set_ylabel('num. of theta\nwaves / 20 s')
		s_i += 1

		data = pd.read_csv(data_occ.format(wave_type))
		if additional_parameter == 'energy':
			histogram_energy = _get_histogram_values_in_sec(data, total_len, 20, 'energy')
			t = histogram_energy['time'] / 3600
			bar_width = t[2] - t[1]
			axs[s_i].bar(t, histogram_energy['occurences'], width=bar_width, color='#b2b2b2', edgecolor='#b2b2b2')
			axs[s_i].set_xlim([t.iloc[0], t.iloc[-1]])
			axs[s_i].set_ylabel('energy [$\mathrm{\mu V^2}$]')
		elif additional_parameter == 'amplitude':
			t = np.linspace(0, total_len, total_len)		
			x = np.zeros(len(t))
			for index, row in data.iterrows():
				idx = np.abs(t -row["absolute_position"]).argmin()
				x[idx] += row["amplitude"]
			t = t / 3600
			axs[s_i].plot(t, x, color='#b2b2b2')
			axs[s_i].set_xlim([t[0], t[-1]])
			axs[s_i].set_ylabel('Amplitude [$\mathrm{\mu V}$]')
			if wave_type == "spindle":
				axs[s_i].set_ylim([0, 150])
			elif wave_type == 'SWA':
				axs[s_i].set_ylim([0, 800])
			else:
				axs[s_i].set_ylim([0, 100])
		s_i += 1
		if s_i==(2 * len(WAVE_TYPES)): axs[s_i-1].set_xlabel('time [hours]')

	if annotate:
		axs[0].annotate('\n'.join(text), xy=(0.3, 1), xytext=(0.7, 0.8), xycoords='figure fraction', fontsize=12)
	py.tight_layout()

	return fig


def _get_histogram_values_in_sec(data, total_len, bin_width, profile_type='amplitude'):
	### profile_type = 'amplitude', 'percent', 'counts', 'energy'

	number_of_bins = int(total_len / bin_width)
	bins = np.zeros(number_of_bins)
	time = np.linspace(0, float(number_of_bins * bin_width) - bin_width, len(bins))

	if profile_type == 'percent':
		for bin_, page in enumerate(time):
			time_page = np.linspace(page, page + bin_width, bin_width * fs)
			x = np.zeros(len(time_page))
			df = data[(data['offset'] > page) & (data['offset'] < (page + bin_width))]
			if len(df):
				for (index, row) in df.iterrows():
					start = np.argmin(np.abs(time_page - float(row["offset"])))
					end = np.argmin(np.abs(time_page - float(row["offset"] + row["struct_len"])))
					if end > time_page[-1]: end = time_page[-1]
					x[int(start):int(end)] = 1
			bins[bin_] = (float(np.sum(x==1)) / len(time_page)) * 100
	else:
		for (index, row) in data.iterrows():
			bin_index = int(np.floor(row['absolute_position'] / bin_width))
			if bin_index == len(time): bin_index = bin_index - 1
			if profile_type == 'amplitude':
				bins[bin_index] += row['amplitude'] 
			elif profile_type == 'counts':
				bins[bin_index] += 1
			elif profile_type == 'energy':
				bins[bin_index] += row['modulus']

	return pd.DataFrame({'time': time, 'occurences': bins})


# def _get_histogram_values(data, total_len, orig_bin_width, profile_type='amplitude'):
# 	### profile_type = 'amplitude', 'percent', 'counts'

# 	total_len = total_len / 3600
# 	# Absolute position is given in hours.
# 	data['absolute_position'] = data['absolute_position']/3600 #((data['book_id']-1)*20. + data['position'])/3600

# 	if profile_type == 'percent':
# 		bin_width = 0.33#1./3
# 	else:
# 		bin_width = orig_bin_width

# 	number_of_bins = int(total_len*60/bin_width+1)
# 	bins = np.zeros(number_of_bins)
# 	time = np.linspace(0, float(number_of_bins*bin_width)/60, len(bins))
# 	for (index, row) in data.iterrows():
# 		if profile_type == 'amplitude':
# 			bins[int(np.floor(row['absolute_position']*60/bin_width))] += row['amplitude'] #row['energy']
# 		elif profile_type == 'counts':
# 			bins[int(np.floor(row['absolute_position']*60/bin_width))] += 1
# 		elif profile_type == 'percent':
# 			bins[int(np.floor(row['absolute_position']*60/bin_width))] += row['struct_len']
# 	if profile_type == 'percent':
# 		bins = (bins/20)*100
# 		bins[np.where(bins>100)[0]] = 100
# 	return pd.DataFrame({'time': time, 'occurences': bins})


def create_histograms(name, directory, orig_bin_width):
	data_name = "{}/{}_{}_occ.csv"
	hipnogram_out_name = "{}/{}".format(directory, name) + "_{}_hipnogram.csv"

	with open(pages_db) as f:
		key = name.split('.')[0]
		pages_data = json.load(f)
		total_len = float(pages_data[key] * 20) # in sec

	for (i, wave_type) in enumerate(WAVE_TYPES):
		data = pd.read_csv(data_name.format(directory, name, wave_type))
		histogram = _get_histogram_values_in_sec(data, total_len, orig_bin_width, PROFILE_TYPES[i])
		histogram.to_csv(hipnogram_out_name.format(wave_type))


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Create histrograms for profiles')
	parser.add_argument('files', nargs='+', metavar='file', help='path to *.csv data files')
	parser.add_argument('--bin-width', type=int, required=True, help='Bin width given in minutes')
	namespace = parser.parse_args()

	for path_to_csv in namespace.files:
		directory = os.path.dirname(path_to_csv)
		file_name = os.path.basename(path_to_csv)
		name = "_".join(file_name.split("_")[:4])
		# create_histograms(name, directory, namespace.bin_width)

		# fig, ss_count, ss_amp, swa_amp = create_combo_plot(name, directory)
		fig = profile_energy_plot(name, directory, additional_parameter='amplitude', annotate=False)
		# fig.suptitle(name, fontsize=20)

		# fig = create_plots(name, directory)

		if not os.path.exists(os.path.join(directory, 'profiles')):
			os.makedirs(os.path.join(directory, 'profiles'))
		py.savefig(os.path.join(directory, 'profiles', name+'_theta_profil_amplitud.png'))
		#py.savefig(os.path.join(directory, 'profiles', name+'_profil_amplitud.eps'), format='eps', dpi=1000)
		py.close(fig)

		# fig = create_amplitude_plots(name, directory, annotate=True)
		# py.savefig(os.path.join(directory, 'profiles', name+'_profil_amplitud_sel.pdf'), papertype = 'a4', format = 'pdf')
		# py.close(fig)
