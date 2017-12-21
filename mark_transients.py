#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse

from book_reader import *
import matplotlib.pyplot as plt
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
from scipy import stats
import shlex

from obci.analysis.obci_signal_processing.tags import tags_file_reader as tags_reader
from obci.analysis.obci_signal_processing.tags import tags_file_writer as tags_writer
from obci.analysis.obci_signal_processing.tags import tag_utils


channels_names_ears_smp = ['Fp1','Fp2','AFz',
						   'F7','F3','Fz','F4','F8',
						   'T3','C3','Cz','C4','T4',
						   'T5','P3','Pz','P4','T6',
						   'O1','O2']    


pages_db = "./pages_db.json"
channels_db = "./channels_db.json"

OUT_DIR = './results'
READBOOK_PATH = '/home/mzieleniewska/empi/empi/utils/readbook'

fs = 128.


def get_rms_percentile(x, freq_range, window_width, percentile, apply_filter=True):
	d, c = ss.butter(2, np.array(freq_range) / (fs / 2), btype='bandpass')
	if apply_filter:
		sig = ss.filtfilt(d, c, x)
	else:
		sig = x
	rms = []
	window = int(window_width * fs)
	for i in xrange(0, sig.shape[0] - window, int(window)):
		s = sig[i:i+window]
		rms.append(np.sqrt(np.mean(np.array(s)**2)))
	rms = np.array(rms)
	r = plt.boxplot(rms, whis=3.5)
	upper_whis = r['whiskers'][1].get_data()[1][1]
	rms_ampli = scoreatpercentile(rms[rms < upper_whis], percentile)*2*np.sqrt(2) 
	return rms, rms_ampli


def get_channel_signal_without_artifacts(signal_channel, ch_id, artifacts_file, artifact_types):
	artifacts = np.zeros(len(signal_channel))
	tags = tags_reader.TagsFileReader(artifacts_file).get_tags()
	for tag in tags:
		if tag['name'] in artifact_types and np.float(tag['channelNumber'])==ch_id:
			artifacts[int(tag['start_timestamp']*fs):int(tag['end_timestamp']*fs)] = 2
	x = np.delete(signal_channel, np.where(artifacts==2))
	# signal_channel[np.where(artifacts==2)] = 0
	return x


def filter_selected_atoms(atoms, ch_id, freq_range, width_range, width_coeff, amplitude_range):
	chosen = []
	for atom in atoms:
		channel = atom[0]
		if channel == ch_id:
			iteration = atom[1]
			modulus   = atom[2]
			amplitude = 2 * atom[3]
			position  = atom[4]
			width     = atom[5]
			frequency = atom[6]
			if (width_range[0] <= width <= width_range[1]) and (amplitude_range[0] <= amplitude <= amplitude_range[1]) and (freq_range[0] <= frequency <= freq_range[1]):
				struct_len = width_coeff * width
				offset = position - struct_len/2
				chosen.append([iteration - 1, modulus, amplitude, width, frequency, struct_len, position, offset])
	df = pd.DataFrame(np.array(chosen), columns=['iteration','modulus','amplitude','width','frequency','struct_len','absolute_position', 'offset'])
	return df


def get_array_with_artifacts_samples(artifact_file, artifact_types, ch_id, fs, nb_samples):
	t = np.zeros(nb_samples)
	tags = tags_reader.TagsFileReader(artifact_file).get_tags()
	for tag in tags:
		if tag['name'] in artifact_types and np.float(tag['channelNumber']) == (ch_id - 1):
			t[int(tag['start_timestamp'] * fs):int(tag['end_timestamp'] * fs)] = 2
	return t


def get_array_with_structures_positions(df, fs, nb_samples):
	t = np.zeros(nb_samples)
	offset = df['offset'] * fs
	width = np.round(df['width'] * fs)
	for i in xrange(len(offset)):
		t[int(offset[i]):int(offset[i] + width[i])] = 1
	return t


def remove_structures_from_df(df, fs, ch_id, nb_samples, artifact_file, artifact_types, overlap):
	t_struct = get_array_with_structures_positions(df, fs, nb_samples)
	t_art = get_array_with_artifacts_samples(artifact_file, artifact_types, ch_id, fs, nb_samples)
	t_fin = t_struct + t_art
	ids = []
	ids.extend(df.index[(df['absolute_position'] <= 20.) | (df['absolute_position'] > (nb_samples / fs) - 20.)].values)
	offset = df['offset'] * fs
	width = np.round(df['struct_len'] * fs)
	df_clean = df.copy(deep=True)
	for i in xrange(len(offset)):
		if overlap != 0:
			if subseq_in_seq(list([3]*int(overlap/100.*width[i])), list(t_fin[int(offset[i]):int(offset[i] + width[i])])) != -1:
				ids.append(i)
		else:
			if 3 in list(t_fin[int(offset[i]):int(offset[i] + width[i])]):
				ids.append(i)
	df_clean.drop(df_clean.index[ids], inplace=True)
	df_clean = df_clean.reset_index()
	return df_clean


def svarog_tags_writer_one_channel(df, f_name, dir_name, fs, ch_id, tag_name):
	out_file = os.path.join(dir_name, f_name + '.tag')
	writer = tags_writer.TagsFileWriter(out_file)
	amp = df['amplitude']
	offset = df['offset']
	width = df['width']
	bid = df['iteration']
	for i in xrange(len(offset)):
		tag = {'channelNumber':ch_id-1, 'start_timestamp':offset[i], 'end_timestamp':offset[i] + width[i], 
			   'name':tag_name, 'desc':{'amplitude':amp[i], 'width':width[i], 'iteration':bid[i]}}
		writer.tag_received(tag)
	writer.finish_saving(0.0)


def main_new_reader(atoms, ch_id, data_dir, out_dir, f_name, structure):

	number_of_channel_in_book = int(atoms[-1, 0])

	#signal = np.fromfile(os.path.join(data_dir, f_name + "_ears_full_128.bin"), dtype='float32').reshape(-1, number_of_channel_in_book).T
	signal = np.fromfile(os.path.join(data_dir, f_name + ".bin"), dtype='float32').reshape(-1, number_of_channel_in_book).T #control
	signal_channel = signal[ch_id-1,:]
	
	artifact_types = [u'muscle', u'slope', u'outlier']
	#artifact_file = os.path.join(data_dir, f_name + "_ears_full_128_artifacts.tag")
	artifact_file = os.path.join(data_dir, f_name + "_artifacts.tag")
	x = get_channel_signal_without_artifacts(signal_channel, ch_id, artifact_file, artifact_types)

	freq_spindle_range = [10, 16]
	freq_swa_range = [0.2, 2]
	freq_theta_range = [4, 8]

	window_width = 0.5

	if structure == 'spindle':
		rms, rms_ampli = get_rms_percentile(x, freq_spindle_range, window_width, 99., apply_filter=True)
		print rms_ampli
		df = filter_selected_atoms(atoms, ch_id, freq_spindle_range, [0.4, np.inf], 1.0, [rms_ampli, 100])
	elif structure == 'SWA':
		rms_ampli = 7*np.sqrt(np.median(x ** 2)) 
		print rms_ampli
		df = filter_selected_atoms(atoms, ch_id, freq_swa_range, [0.5, 6], 1.0, [rms_ampli, 600])
	elif structure == 'theta':
		rms, rms_ampli = get_rms_percentile(x, freq_theta_range, window_width, 97., apply_filter=True)
		print rms_ampli
		df = filter_selected_atoms(atoms, ch_id, freq_theta_range, [0.4, np.inf], 1.0, [rms_ampli, 100])
	
	nb_samples = int(signal.shape[1])

	if os.path.isfile(artifact_file):
		df_clean = remove_structures_from_df(df, fs, ch_id, nb_samples, artifact_file, artifact_types, overlap=0)
		if not os.path.exists(os.path.join(out_dir, 'tags')):
			os.makedirs(os.path.join(out_dir, 'tags'))
		svarog_tags_writer_one_channel(df_clean, f_name+'_'+structure+'_cleaned', os.path.join(out_dir, 'tags'), fs, ch_id, structure)
	else:
		print 'No artifact file detected!'
		df_clean = df
	
	return df_clean


if __name__ == '__main__':

	group = 'control' #patients
	mp_type = 'smp'
	#out_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader'
	out_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/control_99rms_new_reader'

	#data_dir = '/home/mzieleniewska/Coma_sleep_2016/w_projekcie/montage_ears/data/' #wszystkie dane 
	data_dir = '/home/mzieleniewska/Coma_sleep_2016/w_projekcie/montage_ears/data_control/' #channels = Fp1, Fp2, F3, Fz, F4, C3, Cz, C4, P3, Pz, P4 #ch_id = 7

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	structures = ['spindle', 'SWA', 'theta']#'spindle', 'SWA']

	file_list = glob.glob(os.path.join(data_dir, "*.b"))
	
	ch_id = 7 #Cz 11

	for id_f in xrange(0, len(file_list)):
		f = file_list[id_f]
		if group=='patients':
			f_parts = os.path.split(f)[1].split('.')[0].split('_')
			f_name = '_'.join(f_parts[:4])
			extension = 'ears_full_128_' + mp_type
			path_to_txt = os.path.join(data_dir, f_name + '_ears_full_128.txt')
		else:
			f_name = os.path.split(f)[1].split('_')[0]+'_ears'
			extension = mp_type
			path_to_txt = os.path.join(data_dir, f_name + '.txt')
		if not os.path.exists(path_to_txt):
			os.system(READBOOK_PATH + ' ' +  shlex.quote(path_to_book) + ' > ' + shlex.quote(path_to_txt)) #tylko python3!
		atoms = np.loadtxt(path_to_txt)
		for structure in structures:
			print f_name
			df = main_new_reader(atoms, 11, data_dir, out_dir, f_name, structure)
			try:
				df.to_csv(os.path.join(out_dir, 'occ_results', f_name+'_' + structure + '_occ.csv'))
			except IOError:
				os.makedirs(os.path.join(out_dir, 'occ_results'))
