#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse

from book_reader import *
import matplotlib.pyplot as py
import numpy as np
import scipy.signal as ss
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile, gamma
import pandas as pd
import sys
import os, glob
import json
from collections import defaultdict
import mne
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from obci.analysis.obci_signal_processing.tags import tags_file_reader as tags_reader
from obci.analysis.obci_signal_processing.tags import tags_file_writer as tags_writer
from obci.analysis.obci_signal_processing.tags import tag_utils

channels_names_long = ['Fp1-F3','F3-C3','C3-P3','P3-O1',
					   'Fp2-F4','F4-C4','C4-P4','P4-O2',
					   'Fp1-F7','F7-T3','T3-T5','T5-O1',
					   'Fp2-F8','F8-T4','T4-T6','T6-O2',
					   'Fz-Cz','Cz-Pz']

channels_names_ears_smp = ['Fp1','Fp2','AFz',
						   'F7','F3','Fz','F4','F8',
						   'T3','C3','Cz','C4','T4',
						   'T5','P3','Pz','P4','T6',
						   'O1','O2']    

channels_names_ears_mmp = ['Fp1','Fp2',
						   'F3','Fz','F4',
						   'C3','Cz','C4',
						   'P3','Pz','P4']  

pages_db = "./pages_db.json"
channels_db = "./channels_db.json"

OUT_DIR = './results'


def filter_atoms_KC(output_file, epoch_len, corrupted_epochs, ptspmV, atoms, fs, freq_range, width_range, width_coeff, amplitude_range, phase_range):
	f = open(output_file, 'w')
	f.write('book_number,position,modulus,amplitude,width,frequency,offset,struct_len\n')
	chosen = []
	for booknumber in atoms:
		if booknumber not in corrupted_epochs:
			for it,atom in enumerate(atoms[booknumber]):
				if atom['type'] == 13:
					position  = atom['params']['t']/fs
					width     = atom['params']['scale']/fs
					frequency = atom['params']['f']*fs/2
					amplitude = 2*atom['params']['amplitude']/ptspmV
					modulus   = atom['params']['modulus']
					phase     = atom['params']['phase']
					if (width_range[0] <= width <= width_range[1]) and (amplitude_range[0] <= amplitude <= amplitude_range[1]) and (freq_range[0] <= frequency <= freq_range[1]) and (phase <= phase_range[0]  or phase >= phase_range[1]):
						struct_len = round(width_coeff*width*fs)
						struct_pos = position*fs
						if struct_pos < struct_len/2:
							struct_offset = (booknumber-1)*epoch_len
						else:
							struct_offset = (booknumber-1)*epoch_len+struct_pos-round(struct_len/2)
						if struct_pos+struct_len/2>epoch_len:
							struct_len = epoch_len-struct_pos+round(struct_len/2)
						f.write('{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:d},{}\n'.format(booknumber, position, modulus, amplitude, width, frequency, int(struct_offset), struct_len))
						chosen.append([booknumber, position, modulus, amplitude, width, frequency, struct_offset, struct_len])
	f.close()
	return chosen


def filter_atoms(output_file, epoch_len, corrupted_epochs, ptspmV, atoms, fs, freq_range, width_range, width_coeff, amplitude_range):
	f = open(output_file, 'w')
	f.write('book_number,iteration,position,modulus,amplitude,width,frequency,offset,struct_len\n')
	chosen = []
	for booknumber in atoms:
		if booknumber not in corrupted_epochs:
			for it,atom in enumerate(atoms[booknumber]):
				if atom['type'] == 13:
					position  = atom['params']['t']/fs
					width     = atom['params']['scale']/fs
					frequency = atom['params']['f']*fs/2
					amplitude = 2*atom['params']['amplitude']/ptspmV
					modulus   = atom['params']['modulus']
					if (width_range[0] <= width <= width_range[1]) and (amplitude_range[0] <= amplitude <= amplitude_range[1]) and (freq_range[0] <= frequency <= freq_range[1]):
						struct_len = round(width_coeff*width*fs)
						struct_pos = position*fs
						if struct_pos < struct_len/2:
							struct_offset = (booknumber-1)*epoch_len
						else:
							struct_offset = (booknumber-1)*epoch_len+struct_pos-round(struct_len/2)
						if struct_pos+struct_len/2>epoch_len:
							struct_len = epoch_len-struct_pos+round(struct_len/2)
						f.write('{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:d},{}\n'.format(booknumber, it, position, modulus, amplitude, width, frequency, int(struct_offset), struct_len))
						chosen.append([booknumber, it, position, modulus, amplitude, width, frequency, struct_offset, struct_len])
	df = pd.DataFrame(np.array(chosen), columns=['book_id','iteration','position','modulus','amplitude','width','frequency','offset','struct_len'])
	f.close()
	return df

def plot_pages_with_selected_atoms(b, csv_name, channel_name, freq_range, page_id=0):
	atoms = b.atoms
	idx = np.where(np.array(channels_names)==channel_name)
	channel = idx[0][0]+1
	nb_of_pages = len(atoms[channel])
	df = pd.read_csv(csv_name)
	if page_id:
		fig = plot_single_page(b, channel, df, page_id, epoch_len, freq_range)
		# fig.set_size_inches([ 16., 9.55]) #whole screen
		py.show()
	else:
		for i in xrange(nb_of_pages):
			b_id = i+1
			if any(df.book_number == b_id):
				fig = plot_single_page(b, channel, df, b_id, epoch_len, freq_range)
				# fig.set_size_inches([ 16., 9.55]) #whole screen
				py.show()
				raw_input("Press Enter to continue...")
				py.close(fig)
	return fig 

def plot_single_page(b, channel, df, b_id, epoch_len, freq_range):
	print 'page: ', b_id
	signals = b.signals
	structure = df.loc[df['book_number'] == b_id]
	rec = b._reconstruct_page(b_id, channel, freq_range)
	rec_all = b._reconstruct_page(b_id, channel, [0.5, 40])
	sel_gabors = np.zeros(epoch_len)
	for index, row in structure.iterrows():
		sel_gabors += b._gabor(row['amplitude'], row['position'], row['width'], row['frequency'], 0)
		###
		frag_spindle = rec[row['position'] * fs - int(row['width'] * fs / 2) : row['position'] * fs + int(row['width'] * fs / 2)]
		frag = rec_all[row['position'] * fs - int(row['width'] * fs / 2) : row['position'] * fs + int(row['width'] * fs / 2)]
		print 'index ', index, 'energy: ', np.sum(np.abs(frag_spindle) ** 2) / np.sum(np.abs(frag) ** 2)
		###
	
	sig = signals[channel][b_id]
	t = np.linspace(0, 20, epoch_len)
	fig = py.figure()
	py.subplot(311)
	py.plot(t, sig)
	for index, row in structure.iterrows():
		py.axvline(x=row['position']-row['width']/2, ymin=np.min(sig), ymax = np.max(sig), linewidth=1.5, color='red')
		py.axvline(x=row['position']+row['width']/2, ymin=np.min(sig), ymax = np.max(sig), linewidth=1.5, color='red')
	py.subplot(312)
	py.plot(t, rec)
	for index, row in structure.iterrows():
		py.axvline(x=row['position'], ymin=np.min(rec), ymax = np.max(rec), linewidth=1.5, color='red')
	py.subplot(313)
	py.plot(t,sel_gabors)
	return fig

def check_for_artifacts(book, channel, window_width):
	chann_x = book.signals[channel]
	x = np.zeros(len(chann_x) * book.epoch_s)
	for i in xrange(len(chann_x)):
		x[i * book.epoch_s:(i+1) * book.epoch_s] = chann_x[i+1]
	window = int(window_width * book.fs)
	energy = []
	for i in xrange(0, len(x) - window, int(window)):
		energy.append(np.sum(x[i:i+window] ** 2))
	iqr = np.subtract(*np.percentile(energy, [75, 25]))
	energy_thr_upper = np.percentile(energy, 75) + 1.5 * iqr
	# energy_thr_lower = np.percentile(energy, 25) - 1.5 * iqr   
	# corrupted_epochs = np.where(energy > np.percentile(energy, 99.9))[0]
	corrupted_epochs = np.where(energy > energy_thr_upper)[0]
	return energy, np.array(corrupted_epochs)+1

def mark_corrupted_epochs(book, channel):
	s = book.signals[channel]
	energy = np.zeros(len(s))
	diff = np.zeros(len(s))
	for i in xrange(len(s)):
		energy[i] = np.sum(s[i+1] ** 2)
		diff[i] = np.abs(np.max(s[i+1])-np.min(s[i+1]))
	iqr_energy = np.subtract(*np.percentile(energy, [75, 25]))
	energy_thr_upper = np.percentile(energy, 75) + 2.5 * iqr_energy
	iqr_diff = np.subtract(*np.percentile(diff, [75, 25]))
	diff_thr_upper = np.percentile(diff, 75) + 2.5 * iqr_diff
	corrupted_epochs_energy = np.array(np.where(energy > energy_thr_upper)[0])+1
	corrupted_epochs_diff = np.array(np.where(diff > diff_thr_upper)[0])+1
	to_remove = set(corrupted_epochs_energy).difference(corrupted_epochs_diff)
	corrupted_epochs = [ep for ep in corrupted_epochs_energy if ep not in to_remove]
	return corrupted_epochs

def get_rms_values(book, channel, rms_type, freq_spindle_range, window_width):
	chann_x = book.signals[channel]
	x = np.zeros(len(chann_x) * book.epoch_s)
	for i in xrange(len(chann_x)):
		x[i * book.epoch_s:(i+1) * book.epoch_s] = chann_x[i+1]
	if rms_type == 'filter_signal':
		fn = book.fs / 2.
		d, c = ss.butter(2, np.array(freq_spindle_range) / fn, btype='bandpass')
		sig = ss.filtfilt(d, c, x)
	elif rms_type == 'filter_recon':
		sig = book._reconstruct_signal(channel, freq_spindle_range)
	window = int(window_width * book.fs)
	rms = []
	energy, corrupted_epochs = check_for_artifacts(book, channel, window_width)
	for i in xrange(0, len(sig) - window, int(window)):
		s = sig[i:i+window]
		rms.append(np.sqrt(np.mean(np.array(s)**2)))
	rms = np.array(rms)
	cleared_rms = np.delete(rms, corrupted_epochs)
	return cleared_rms, sig, corrupted_epochs

def get_rms_multiple_channels(book, selected_channels, channels_names, freq_spindle_range, window_width, percentile, apply_filter=True):
	signals = book.signals
	for i,ch in enumerate(selected_channels):
		idx = np.where(np.array(channels_names)==ch)
		channel = idx[0][0]+1
		corrupted_epochs = [] #mark_corrupted_epochs(book, channel)
		for booknumber in signals[channel]:
			if booknumber not in corrupted_epochs:
				try:
					x = np.hstack((x, signals[channel][booknumber]))
				except NameError:
					x = signals[channel][booknumber]
	d, c = ss.butter(2, np.array(freq_spindle_range) / (book.fs / 2), btype='bandpass')
	if apply_filter:
		sig = ss.filtfilt(d, c, x)
	else:
		sig = x
	rms = []
	window = int(window_width * book.fs)
	for i in xrange(0, len(sig) - window, int(window)):
		s = sig[i:i+window]
		rms.append(np.sqrt(np.mean(np.array(s)**2)))
	rms = np.array(rms)
	rms_ampli = scoreatpercentile(rms, percentile)*2*np.sqrt(2) 
	return rms, rms_ampli

def get_whole_signal(signals, selected_channels, epoch_len, number_of_epochs):
	signal = np.zeros((len(selected_channels), epoch_len * number_of_epochs))
	for i, ch in enumerate(selected_channels):
		flag = 0
		idx = np.where(np.array(channels_names) == ch)
		channel = idx[0][0] + 1
		page_sig = signals[channel]
		for p in page_sig.keys():
			signal[i, flag : flag + epoch_len] = page_sig[p] 
			flag = flag + epoch_len
	return signal

def spectrum_model(f, alpha, b):
	return b + 1. / (f ** alpha)

def select_channel(b, channels_names, selected_channels, freq_range):
	channels_energy = np.zeros(len(selected_channels))
	for ch_id, ch in enumerate(selected_channels):
		channel = np.where(np.array(channels_names)==ch)[0][0]+1
		reconstruction_in_range = b._reconstruct_signal(channel, freq_range)
		channels_energy[ch_id] = np.sum(reconstruction_in_range**2)
	sel_id = np.argmax(channels_energy)
	channel_name = selected_channels[sel_id]
	channel_id = np.where(np.array(channels_names)==channel_name)[0][0]+1
	return channel_name, channel_id

def select_rhythm(b, channels_names, selected_channels, fs, freq_range, window_width, min_width, rms_percentile, out_dir, f_name, structure):
	channel_name, channel_id = select_channel(b, channels_names, selected_channels, freq_range)
	patient_id = f_name.split('_')[0]
	ids = patient_id + '_' + structure
	try:
		with open(channels_db) as f:
			channels_data = json.load(f)
		if ids in channels_data.keys():
			channel_name = channels_data[ids]
		else:
			channels_data[ids] = channel_name
			with open(channels_db, 'w') as f_out:
				json.dump(channels_data, f_out)
	except IOError:
		with open(channels_db, 'w') as f:
			json.dump({ids: channel_name}, f)
	corrupted_epochs = mark_corrupted_epochs(b, channel_id)
	rms, rms_ampli = get_rms_multiple_channels(b, [channel_name], channels_names, freq_range, window_width, rms_percentile)
	chosen = filter_atoms(os.path.join(out_dir, f_name + '_' + channel_name + '_' + structure + '.csv'), b.epoch_s, corrupted_epochs, b.ptspmV, b.atoms[channel_id], fs, freq_range, [min_width, np.inf], 1.0, [rms_ampli, np.inf])
	return chosen

def svarog_tags_writer(df, f_name, dir_name, fs, selected_channels, all_channels):
	out_file = os.path.join(dir_name, f_name + '.tag')
	writer = tags_writer.TagsFileWriter(out_file)
	for ch_id in df.keys():
		amp = df[ch_id]['amplitude']
		offset = df[ch_id]['offset']
		width = df[ch_id]['width']
		ch_id_real = ch_id #all_channels.index(selected_channels[ch_id-1])
		for i in xrange(len(offset)):
			tag = {'channelNumber':ch_id_real, 'start_timestamp':offset[i]/fs, 'end_timestamp':offset[i]/fs+width[i], 
				   'name':'spindles', 'desc':{'amplitude':amp[i], 'width':width[i]}}
			writer.tag_received(tag)
	writer.finish_saving(0.0)

def plot_topography(x, channels_names, fs):
	info = mne.create_info(channels_names, fs, 'eeg', 'standard_1020')
	layout = mne.channels.make_eeg_layout(info)
	axs, cont = mne.viz.plot_topomap(x, layout.pos)
	fig = py.gcf()
	return fig, axs, cont

def plot_averaged_topography(df, atoms, fs, freq_range):
	amps, freqs = get_amplitudes_in_channels(df, atoms)
	channel_names = channels_names_ears_mmp #['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
	info = mne.create_info(channel_names, fs, 'eeg', 'standard_1020')
	layout = mne.channels.make_eeg_layout(info)
	ids = np.where((freqs>=freq_range[0]) & (freqs<freq_range[1]))[0]
	topo = np.mean(amps[ids,:], 0)
	mne.viz.plot_topomap(topo, layout.pos)
	fig = py.gcf()
	return fig

def get_amplitudes_in_channels(df, atoms):
	found = []
	amps = []
	freqs = []
	for ch_id in df.keys():
		offset = df[ch_id]['offset']
		amplitude = df[ch_id]['amplitude']
		frequency = df[ch_id]['frequency']
		book_id = df[ch_id]['book_id']
		iteration = df[ch_id]['iteration']
		for i in xrange(len(offset)):
			if offset[i] not in found:
				amp_temp = []
				for a in atoms.keys():
					a_chann = atoms[a]
					amp_temp.append(a_chann[int(book_id[i])][int(iteration[i])]['params']['amplitude'])
				amps.append(amp_temp)
				found.append(offset[i])
				freqs.append(frequency[i])
	return np.array(amps), np.array(freqs)

def get_time_array_with_positions(df, fs, epoch_len, number_of_epochs):
	t = np.zeros(number_of_epochs*epoch_len)
	for ch_id in df.keys():
		offset = np.round(df[ch_id]['offset'])
		width = np.round(df[ch_id]['width']*fs)
		for i in xrange(len(offset)):
			t[int(offset[i]):int(offset[i]+width[i])] = 1
	return t

def get_time_array_with_artifacts(f_name, artifact_types, fs, epoch_len, number_of_epochs):
	t = np.zeros(number_of_epochs*epoch_len)
	tags = tags_reader.TagsFileReader(f_name).get_tags()
	for tag in tags:
		if tag['name'] in artifact_types:
			t[int(tag['start_timestamp']*fs):int(tag['end_timestamp']*fs)] = 2
	return t

def remove_structures(df, fs, epoch_len, number_of_epochs, artifact_file, artifact_types, overlap):
	t_struct = get_time_array_with_positions(df, fs, epoch_len, number_of_epochs)
	t_art = get_time_array_with_artifacts(artifact_file, artifact_types, fs, epoch_len, number_of_epochs)
	t_fin = t_struct + t_art
	df_clean = defaultdict(list)
	for ch_id in df.keys():
		offset = np.round(df[ch_id]['offset'])
		width = np.round(df[ch_id]['width']*fs)
		ids = []
		df_clean[ch_id] = df[ch_id].copy(deep=True)
		for i in xrange(len(offset)):
			if subseq_in_seq(list([3]*int(overlap/100.*width[i])), list(t_fin[int(offset[i]):int(offset[i]+width[i])])) != -1:
				ids.append(i)
		df_clean[ch_id].drop(df_clean[ch_id].index[ids], inplace=True)
		df_clean[ch_id] = df_clean[ch_id].reset_index()
	return df_clean

def subseq_in_seq(subseq, seq):
	i, n, m = -1, len(seq), len(subseq)
	try:
		while True:
			i = seq.index(subseq[0], i + 1, n - m + 1)
			if subseq == seq[i:i + m]:
			   return i
	except ValueError:
		return -1

def gaussian_fun(t, sigma1, mu1, amp1, sigma2, mu2, amp2):
	p1 = amp1 * np.exp(-(t - mu1)**2 / (2 * sigma1**2))
	p2 = amp2 * np.exp(-(t - mu2)**2 / (2 * sigma2**2))
	return p1 + p2

def plot_histogram(x, number_of_epochs, bin_len):
	total_len = float(number_of_epochs*20.)
	binBoundaries = np.linspace(0, total_len, total_len/bin_len+1)
	vals = py.hist(x, bins=binBoundaries, fc='gray', histtype='stepfilled', alpha=0.3, normed=False)
	return vals[0], vals[1]

def fit_gaussian_mixture_model(data, bins):
	X = data[:, np.newaxis]
	bins = bins[:, np.newaxis]
	N = np.arange(1, 11)
	models = [None for i in range(len(N))]
	for i in range(len(N)):
		models[i] = GaussianMixture(N[i]).fit(X)
	# compute the AIC and the BIC
	AIC = [m.aic(X) for m in models]
	BIC = [m.bic(X) for m in models]

	fig = py.figure()
	ax = fig.add_subplot(121)
	min_nb_comp = np.min([np.argmin(BIC),np.argmin(AIC)])
	M_best = models[min_nb_comp]
	log_prob = M_best.score_samples(bins)
	pdf = np.exp(log_prob)
	ax.hist(X[:,0], 50, normed=True, histtype='stepfilled', alpha=0.4)
	ax.plot(bins, pdf, '-k')
	ax.text(0.04, 0.96, "Best-fit mixture with %s components"%(min_nb_comp+1),
			ha='left', va='top', transform=ax.transAxes)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$p(x)$')
	ax = fig.add_subplot(122)
	ax.plot(N, AIC, '-k', label='AIC')
	ax.plot(N, BIC, '--k', label='BIC')
	ax.set_xlabel('nb of components')
	ax.set_ylabel('information criterion')
	ax.legend(loc=2)
	# py.show()
	return fig

def compute_kmeans_clustering(X, K_MAX):
	from scipy.cluster.vq import kmeans
	from scipy.spatial.distance import cdist, pdist
	from matplotlib import cm

	X = X.reshape(-1, 1)

	n = range(1, K_MAX+1)

	KM = [kmeans(X,k) for k in n]
	centroids = [cent for (cent,var) in KM]

	D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
	cIdx = [np.argmin(D, axis=1) for D in D_k]
	dist = [np.min(D, axis=1) for D in D_k]

	tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
	totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
	betweenss = totss - tot_withinss          # The between-cluster sum of squares

	# kIdx = 9        # K=10
	# elbow curve
	fig = py.figure()
	ax = fig.add_subplot(111)
	ax.plot(n, betweenss/totss*100, 'b*-')
	# ax.plot(n[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, 
	#     markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
	ax.set_ylim((0,100))
	py.grid(True)
	py.xlabel('Number of clusters')
	py.ylabel('Percentage of variance explained (%)')
	py.title('Elbow for KMeans clustering')
	return fig

def compute_pseudo_f(X, K_MAX):
	from sklearn.cluster import KMeans
	from sklearn import metrics

	X = X.reshape(-1, 1)
	n = range(2, K_MAX+1)
	KM = [KMeans(n_clusters=k, random_state=1).fit(X) for k in n]
	pseudo_f = [metrics.calinski_harabaz_score(X, k.labels_) for k in KM]

	fig = py.figure()
	ax = fig.add_subplot(111)
	ax.plot(n, pseudo_f, 'b*-')
	py.xlabel('Number of clusters')
	py.ylabel('pseudo-F index')
	return fig

def compute_autocorrelation(x):
	autocorr = np.correlate(x, x, mode='full')
	autocorr = autocorr[int(autocorr.size/2):]
	return autocorr

def analyse_multichannel_amplitues():
	#################### TODO
	amps, freqs = get_amplitudes_in_channels(df_clean, atoms)
	amps_t = amps.transpose()
	U, s, V = np.linalg.svd(amps_t, full_matrices=True)

	fig = py.figure()
	for c in xrange(U.shape[1]):
		py.subplot(3,4,c+1)
		plot_topography(U[:,c], channels_names, fs)
		py.title(str(c))
	py.show()
	py.savefig(os.path.join(OUT_DIR, f_name+'_svd.png'))
	py.close()

	x = np.diff(np.sort(df[11][df[11]['frequency']>=13]['offset']/fs))
	fit_alpha, fit_loc, fit_beta = gamma.fit(x) #if alpha = 1 poisson, random
	print fit_alpha, fit_loc, fit_beta

	fig = plot_averaged_topography(df_clean, atoms, fs, [12,14])
	fig.savefig(os.path.join(OUT_DIR, f_name+'_slow.png'))
	py.close(fig)
	fig = plot_averaged_topography(df_clean, atoms, fs, [14,16])
	fig.savefig(os.path.join(OUT_DIR, f_name+'_fast.png'))
	py.close(fig)

def analyse_frequency_distribution():
	#################### TODO
	fig = py.figure()
	py.plot(df_clean[11]['frequency'],df_clean[11]['amplitude'],'r.')
	py.plot(df_clean[5]['frequency'],df_clean[5]['amplitude'],'b*')

	f5 = df_clean[5]['frequency']
	f11 = df_clean[11]['frequency']
	f_fin = np.hstack((f5, f11))
	fig = py.figure()
	n, bins, patches = py.hist(f_fin, 50)
	new_bins = []
	for b in xrange(len(bins)-1):
		new_bins.append((bins[b]+bins[b+1])*0.5)
	new_bins = np.array(new_bins)

	ids = ss.argrelextrema(n, np.greater)[0]
	f1 = new_bins[(np.where(n[new_bins<13] == max(n[ids[new_bins[ids]<13]])))][0]
	amp1 = n[new_bins==f1][0]
	f2 = new_bins[(np.where(n[new_bins>=13] == max(n[ids[new_bins[ids]>=13]]))[0]+len(n[new_bins<13]))[0]]
	amp2 = n[new_bins==f2][0]

	# pdf = gaussian_fun(new_bins, mu1, f1, amp1, mu2, f2, amp2)
	popt, pcov = curve_fit(gaussian_fun, new_bins, n, p0=[0.2, f1, amp1, 0.2, f2, amp2])
	fig = py.figure()
	py.plot(new_bins, n, 'b')
	py.plot(new_bins, gaussian_fun(new_bins, *popt), 'g--')


def save_total_number_of_epochs(f_name, number_of_epochs):
	try:
		with open(pages_db) as f:
			pages_data = json.load(f)
		if f_name in pages_data.keys():
			pass
		else:
			pages_data[f_name] = number_of_epochs
			with open(pages_db, 'w') as f_out:
				json.dump(pages_data, f_out)
	except IOError:
		with open(pages_db, 'w') as f:
			json.dump({f_name: number_of_epochs}, f)

def main(data_dir, f_name, channels_names, selected_channels, extension):
	artifact_file = os.path.join(data_dir, 'data', f_name+'_artifacts.tag')
	patient = f_name 
	patient_id = patient.split('_')[0]
	out_dir = os.path.join(data_dir, patient)
	if not os.path.isdir(out_dir):
		os.mkdir(out_dir)
	print 'Working on patient: ', patient
	b = BookImporter(os.path.join(data_dir, 'decomposed_books', f_name+'_'+extension+'.b'))
	atoms = b.atoms
	signals = b.signals
	epoch_len = b.epoch_s
	fs = b.fs
	ptspmV = b.ptspmV
	number_of_epochs = len(signals[1])
	## save total number of epochs for each recording
	save_total_number_of_epochs(f_name, number_of_epochs)

	window_width = 0.2
	freq_spindle_range = [10, 16] #docelowo [9, 16]

	df = defaultdict(list)
	# for ch_id in atoms.keys():
	for channel_name in selected_channels:
		ch_id = channels_names.index(channel_name)
		# channel_name = channels_names[ch_id-1]
		rms_ss, rms_ampli_ss = get_rms_multiple_channels(b, [channel_name], channels_names, freq_spindle_range, window_width, 99.0, apply_filter=True)
		print channel_name, ' ', rms_ampli_ss
		df[ch_id] = filter_atoms(os.path.join(out_dir, f_name + '_' + channel_name + '_SS.csv'), epoch_len, [], ptspmV, atoms[ch_id], fs, freq_spindle_range, [0.30, np.inf], 1.0, [rms_ampli_ss, np.inf])
		df[ch_id]['absolute_position'] = ((df[ch_id]['book_id']-1)*20. + df[ch_id]['position'])

	artifact_types = [u'muscle']
	df_clean = remove_structures(df, fs, epoch_len, number_of_epochs, artifact_file, artifact_types, 20.)
	svarog_tags_writer(df_clean, f_name+'_spindles_cleaned', os.path.join(data_dir, 'data'), fs, selected_channels, channels_names)

	fig = py.figure()
	counts, bins = plot_histogram(df_clean[ch_id]['absolute_position'], number_of_epochs, 180) #3-min bins
	py.savefig(os.path.join(out_dir, patient+'_hist.png'))
	fig.clf()

	new_bins = []
	for b in xrange(len(bins)-1):
		new_bins.append((bins[b]+bins[b+1])*0.5)
	new_bins = np.array(new_bins)
	
	### fit Gaussian Mixture Model with optimal nb of components (AIC or BIC criterion)
	fig = fit_gaussian_mixture_model(df_clean[ch_id]['absolute_position'], new_bins)
	py.savefig(os.path.join(out_dir, patient+'_gmm.png'))
	fig.clf()

	### plot autocorrelation
	autocorr = compute_autocorrelation(counts)
	fig = py.figure()
	py.plot(new_bins, autocorr)
	py.savefig(os.path.join(out_dir, patient+'_autocorr.png'))
	fig.clf()

	### fit Gamma distribution to inter-spindle intervals 
	x_diff = np.diff(np.sort(df_clean[ch_id]['absolute_position']))
	py.figure()
	py.hist(x_diff, 100)
	py.savefig(os.path.join(out_dir, patient+'_intervals.png'))
	fig.clf()

	fig = compute_pseudo_f(df_clean[ch_id]['absolute_position'], 10)
	py.savefig(os.path.join(out_dir, patient+'_km_pseudo_f.png'))
	fig.clf()

	fit_alpha, fit_loc, fit_beta = gamma.fit(x_diff) #if alpha = 1 poisson, random
	print 'alpha, loc, beta:', fit_alpha, fit_loc, fit_beta
	py.close('all')


if __name__ == '__main__':

	# data_dir = '/home/mzieleniewska/empi/from_hpc/data/mmp2/patients/'
	data_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/control/'

	file_list = glob.glob(os.path.join(data_dir, 'decomposed_books', "*.b"))
	channels_names = ['Fp1','Fp2','F3','Fz','F4','C3','Cz','C4','P3','Pz','P4']
	selected_channels = ['C3']
	for f in file_list:
		f_name = os.path.split(f)[1].split('_')[0]
		main(data_dir, f_name+'_ears', channels_names, selected_channels, 'smp')
		
	# recon = b._reconstruct_signal(channel_id, [0.5, 40])
	# recon_spindle = b._reconstruct_signal(channel_id, freq_spindle_range)
