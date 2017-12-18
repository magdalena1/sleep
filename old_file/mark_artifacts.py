#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib.pyplot as py
import numpy as np
import scipy.signal as ss
import sys
import os.path
import glob

from obci.analysis.obci_signal_processing.read_manager import ReadManager

from obci.analysis.obci_signal_processing.tags import tags_file_reader as tags_reader

from artifacts_mark import detect_artifacts, ArtTagsSaver

channels_names_ears = ['Fp1','Fp2','AFz',
					   'F7','F3','Fz','F4','F8',
					   'T3','C3','Cz','C4','T4',
					   'T5','P3','Pz','P4','T6',
					   'O1','O2']  

def get_channels_indices(channels, selected_channels):
	ids = []
	for ch in selected_channels:
		ids.append(np.where(np.array(channels)==ch)[0][0])
	return ids

def get_microvolt_samples(rm, channel=None):
	'''Does get_samples on smart tag (read manager), but multiplied by channel gain'''
	if not channel: #returns for all channels
		gains = np.array([float(i) for i in rm.get_param('channels_gains')], ndmin=2).T
		return rm.get_samples()*gains
	else: #returns for specific channel
		chN = rm.get_param('channels_names').index(channel)
		gain = rm.get_param('channels_gains')[chN]
		return rm.get_channel_samples(channel)*float(gain)

def montage_ears(rm, l_ear_channel, r_ear_channel, exclude_from_montage=[]):
	include_channels = [ch for ch in rm.get_param('channels_names') if ch not in exclude_from_montage]
	x_ref = rm.get_channels_samples(include_channels)-0.5*(rm.get_channel_samples(l_ear_channel)+rm.get_channel_samples(r_ear_channel))
	return np.vstack((x_ref, rm.get_channels_samples(exclude_from_montage)))

def main(data_dir, f_name, selected_channels, preprocess=True):
	try:
		eeg_rm = ReadManager(os.path.join(data_dir, f_name+'.xml'), os.path.join(data_dir, f_name+'.raw'), os.path.join(data_dir, f_name+'.tag'))
	except IOError:
		eeg_rm = ReadManager(os.path.join(data_dir, f_name+'.xml'), os.path.join(data_dir, f_name+'.bin'), os.path.join(data_dir, f_name+'.tag'))
	fs = float(eeg_rm.get_param('sampling_frequency'))

	if preprocess:
		eeg_rm.set_samples(get_microvolt_samples(eeg_rm), eeg_rm.get_param('channels_names'))
		try:
			drop_chnls = [u'EOGL',u'EOGR',u'EMG',u'EKG',u'RESP_U',u'RESP_D',u'TERM',u'SaO2',u'Pleth',u'HRate',u'Saw',u'Driver_Saw']
			x_ref = montage_ears(eeg_rm, 'A1', 'A2', exclude_from_montage=drop_chnls)
		except ValueError:
			drop_chnls = [u'EOGL',u'EOGR',u'EMG',u'EKG',u'RESP_UP',u'RESP_DOWN',u'TERM',u'SaO2',u'Pleth',u'HRate',u'Saw',u'Driver_Saw']
			x_ref = montage_ears(eeg_rm, 'A1', 'A2', exclude_from_montage=drop_chnls)
		eeg_rm.set_samples(x_ref, eeg_rm.get_param('channels_names'))

		# [b_stop, a_stop] = ss.butter(4, np.array([37, 39])/(fs/2), btype='stop', analog=0, output='ba')
		# [b_notch, a_notch] = ss.butter(4, np.array([49.5, 50.5])/(fs/2), btype='stop', analog=0, output='ba')
		# [b_low, a_low] = ss.butter(5, 45.0/(fs/2), btype='low', analog=0, output='ba')
		# [b_high, a_high] = ss.butter(4, 0.5/(fs/2), btype='high', analog=0, output='ba')
		# s = eeg_rm.get_channels_samples(selected_channels)
		# x = ss.filtfilt(b_high, a_high, s) 
		# x = ss.filtfilt(b_low, a_low, x)
		# x = ss.filtfilt(b_stop, a_stop, x)

	available_chnls = eeg_rm.get_param('channels_names')
	Channels2BeTested = [c for c in selected_channels if c in available_chnls]
	AllChannels = available_chnls
	#filtrowanie dla zwykłych artefaktów:
	filters = [[0.5, 0.25, 3, 6.97, "butter"],
		   [30, 60, 3, 12.4, "butter"],
		   [[47.5,   52.5], [ 49.9,  50.1], 3, 25, "cheby2"]] 

	#filtrowanie dla detekcji mięśni:
	filters_muscle = [[0.5, 0.25, 3, 6.97, "butter"],
			  [128, 192, 3, 12, "butter"]]
	line_fs = [50., 100., 150, 200]
	for line_f in line_fs: 
		filters_muscle.append([[line_f-2.5, line_f+2.5], [line_f-0.1, line_f+0.1], 3.0, 25., "cheby2"])

	ArtDet_kargs = {}

	#jeśli którys z poniższych parametrów nie został podany ręcznie, to jest używana wartość domyślna
	forget = ArtDet_kargs.get('forget', 2.) # [s] długość fragmentów na początku i końcu sygnału, dla których nie zapisujemy tagów

	SlopeWindow = ArtDet_kargs.get('SlopeWindow', 0.07)       # [s] #szerokość okna, w którym szukamy iglic i stromych zboczy (slope)
	SlowWindow  = ArtDet_kargs.get('SlowWindow', 1)          # [s] #szerokość okna, w którym szukamy fal wolnych (slow)
	OutliersMergeWin =ArtDet_kargs.get('OutliersMergeWin', .1)# [s] #szerokość okna, w którym łączymy sąsiadujące obszary z próbkami outlier
	MusclesFreqRange = ArtDet_kargs.get('MusclesFreqRange', [40,250]) # [Hz] przedział częstości, w którym szukamy artefaktów mięśniowych

	#progi powyżej których próbka jest oznaczana, jako slope:
	SlopesAbsThr = ArtDet_kargs.get('SlopesAbsThr', np.inf)       # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlopeWindow
	SlopesStdThr = ArtDet_kargs.get('SlopesStdThr', np.inf)        # wartość peak2peak w oknie, jako wielokrotność std
	SlopesThrRel2Med = ArtDet_kargs.get('SlopesThrRel2Med', np.inf)# wartość peak2peak w oknie, jako wielokrotność mediany
	SlopesThrs = (SlopesAbsThr, SlopesStdThr, SlopesThrRel2Med)

	#progi powyżej których próbka jest oznaczana, jako slow (fala wolna):
	SlowsAbsThr = ArtDet_kargs.get('SlowsAbsThr', np.inf)       # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlowWindow
	SlowsStdThr = ArtDet_kargs.get('SlowsStdThr', np.inf)        # wartość peak2peak w oknie, jako wielokrotność std
	SlowsThrRel2Med = ArtDet_kargs.get('SlowsThrRel2Med', np.inf)# wartość peak2peak w oknie, jako wielokrotność mediany
	SlowsThrs = (SlowsAbsThr, SlowsStdThr, SlowsThrRel2Med)

	#progi powyżej których próbka jest oznaczana, jako outlier:
	OutliersAbsThr = ArtDet_kargs.get('OutliersAbsThr', 700)       # [µV] bezwzględna wartość amplitudy (liczona od 0)
	OutliersStdThr = ArtDet_kargs.get('OutliersStdThr', 22)         # wartość amplitudy, jako wielokrotność std
	OutliersThrRel2Med = ArtDet_kargs.get('OutliersThrRel2Med', np.inf)# wartość amplitudy, jako wielokrotność mediany
	OutliersThrs = (OutliersAbsThr, OutliersStdThr, OutliersThrRel2Med)

	#progi powyżej których próbka jest oznaczana, jako muscle:
	MusclesAbsThr = ArtDet_kargs.get('MusclesAbsThr', np.inf)   # [µV²] bezwzględna wartość średniej mocy na próbkę
																# w zakresie częstości MusclesFreqRange, której lepiej nie ustawiać,
																# bo pod tym wzlęgem kanały się bardzo różnią
	MusclesStdThr = ArtDet_kargs.get('MusclesStdThr', 16)        # wartość amplitudy, jako wielokrotność std
	MusclesThrRel2Med = ArtDet_kargs.get('MusclesThrRel2Med', 22)# wartość amplitudy, jako wielokrotność mediany
	MusclesThrs = (MusclesAbsThr, MusclesStdThr, MusclesThrRel2Med)

	# UWAGA! progi odnoszące się do std lub mediany są szacowane dla każdego kanału osobno
	# tagi są zaznaczane wg najostrzejszego z kryteriów
	# żeby nie korzystać z danego kryterium wystarczy podać wartość np.inf
	# każdy tag ma zapisane informacje:
	#   które kryterium było wykorzystane (typ:'abs', 'std' lub 'med')
	#   jaki był ostateczny próg (thr:wartość) w µV
	#   jaka wartość przekroczyła próg (val:wartość) w µV

	tags = detect_artifacts(eeg_rm, Channels2BeTested, AllChannels, filters, filters_muscle, forget, OutliersMergeWin, SlopeWindow, SlowWindow, MusclesFreqRange, SlopesThrs, SlowsThrs, OutliersThrs, MusclesThrs)

	ArtifactsFilePath = os.path.join(data_dir, f_name+'_artifacts.tag')
	ArtTagsSaver(ArtifactsFilePath, tags, '')
	return tags


if __name__ == '__main__':

	data_dir = '/home/mzieleniewska/empi/from_hpc/data/smp/control/data_dir'
	selected_channels = ['Fp1','Fp2','F3','Fz','F4','C3','Cz','C4','P3','Pz','P4']

	file_list = glob.glob(os.path.join(data_dir, "*.bin"))

	for f in file_list:
		f_name = os.path.split(f)[1].split('.')[0]
		tags = main(data_dir, f_name, selected_channels, preprocess=False)

		

