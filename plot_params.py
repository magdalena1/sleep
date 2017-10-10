#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib.pylab as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import glob
import os
import datetime
from dateutil.parser import parse

from collections import defaultdict, OrderedDict

WAVE_TYPES = ['SS','SWA']
crs_file = '/home/mzieleniewska/sleep2/books/crs.csv'


def plot_parameter(vals, param):
	vals.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%d-%m-%Y'))
	fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))
	plt.setp(axs, xticks=range(len(vals)))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		to_plot = []
		to_plot_err = []
		labels = []
		x = np.array(range(0, len(vals)))
		for j,v in enumerate(vals):
			to_plot.append(v[wave_type+'_'+param])
			to_plot_err.append(v[wave_type+'_'+param+'_std'])
			labels.append(v['crs'])			
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, to_plot) #linear regression
		axs[i].plot(x, intercept + slope*x, 'r')
		axs[i].text(0.85, 0.1, 'p = ' +  str(np.round(p_value, 3)), transform=axs[i].transAxes)
		axs[i].errorbar(x, to_plot, yerr=to_plot_err, fmt='o')
		axs[i].set_xlim([-0.25, len(to_plot)-0.75])
		axs[i].set_ylabel('Amplitude [$\mu V$]')
		axs[i].set_title(wave_type)
		axs[i].set_xticklabels(labels)
	return fig

def plot_boxplots(data, patient):
	labels = get_crs(crs_file, patient)
	fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		d = data[patient][wave_type]
		sorted_d = OrderedDict(sorted(d.items(), key=lambda x: parse(x[0])))
		to_boxplot = []
		for k in sorted_d.keys():
			to_boxplot.append(sorted_d[k])
		axs[i].boxplot(to_boxplot)
		axs[i].set_ylabel('Amplitude [$\mu V$]')
		axs[i].set_title(wave_type)
		axs[i].set_xticklabels(labels)
	return fig

def plot_median_with_regression(data, patient, parameter):
	labels = get_crs(crs_file, patient)
	fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		d = data[patient][wave_type]
		x = np.array(range(0, len(d.keys())))
		sorted_d = OrderedDict(sorted(d.items(), key=lambda x: parse(x[0])))
		med = []
		q1 = []
		q3 = []
		for k in sorted_d.keys():
			med.append(np.median(sorted_d[k][parameter]))
			q1.append(np.percentile(sorted_d[k][parameter], 25.0))
			q3.append(np.percentile(sorted_d[k][parameter], 75.0))
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, med)
		axs[i].plot(x, intercept + slope*x, 'g')
		axs[i].errorbar(x, med, yerr=[np.array(med)-np.array(q1), np.array(q3)-np.array(med)], fmt='o')
		axs[i].text(0.85, 0.1, 'p = ' +  str(np.round(p_value, 3)), transform=axs[i].transAxes)
		axs[i].set_ylabel('Amplitude [$\mu V$]')
		axs[i].set_title(wave_type)
		axs[i].set_xticks(x)
		axs[i].set_xticklabels(labels)
		axs[i].set_xlim([-0.25, x.max()+0.25])
	return fig

# def plot_median_with_regression(data, patient, wave_type):
# 	labels = get_crs(crs_file, patient)

# 	d = data[patient][wave_type]
# 	x = np.array(range(0, len(d.keys())))
# 	sorted_d = OrderedDict(sorted(d.items(), key=lambda x: parse(x[0])))
# 	med = []
# 	q1 = []
# 	q3 = []
# 	for k in sorted_d.keys():
# 		med.append(np.median(sorted_d[k]))
# 		q1.append(np.percentile(sorted_d[k], 25.0))
# 		q3.append(np.percentile(sorted_d[k], 75.0))
# 	fig = plt.figure()
# 	slope, intercept, r_value, p_value, std_err = stats.linregress(x, med)
# 	plot(x, intercept + slope*x, 'g')
# 	errorbar(x, med, yerr=[np.array(med)-np.array(q1), np.array(q3)-np.array(med)], fmt='o')
# 	text(0.85, 0.1, 'p = ' +  str(np.round(p_value, 3)), transform=axs[i].transAxes)
# 	set_ylabel('Amplitude [$\mu V$]')
# 	set_title(wave_type)
# 	set_xticks(x)
# 	set_xticklabels(labels)
# 	set_xlim([-0.25, x.max()+0.25])
# 	return fig

def get_crs(crs_file, patient):
	f = pd.read_csv(crs_file)
	p = f.ix[(f['patient_id']==patient)]
	p['date'] = pd.to_datetime(p.date)
	return p.crs

def get_ordered_data(book_dir, file_list):
	patients_list = set()
	for f in file_list:
		patient = os.path.basename(f).split('.')[0]
		patients_list.update([patient.split('_')[0]])
	rec_d = lambda: defaultdict(rec_d)
	d = rec_d()
	for p in patients_list:
		matching = [s for s in file_list if p in s]
		for match in matching:
			patient = os.path.basename(match).split('.')[0]
			patient_dir = os.path.join(book_dir, patient)
			date = patient.split('_')[3]+'-'+patient.split('_')[2]+'-'+patient.split('_')[1]
			for wave_type in WAVE_TYPES:
				structure_file = "{}/{}".format(patient_dir, patient) + "_" + wave_type + ".csv"
				data = pd.read_csv(structure_file)
				data['absolute_position'] = ((data['book_number']-1)*20. + data['position'])
				d[p][wave_type][date] = data
	return d


if __name__ == '__main__':

	out_dir = '/home/mzieleniewska/sleep2/results/params/median'
	book_dir = '/home/mzieleniewska/sleep2/books/one_channel'

	file_list = glob.glob(os.path.join(book_dir, "*.b"))

	data = get_ordered_data(book_dir, file_list)

	for patient in data.keys():
		fig = plot_median_with_regression(data, patient, 'width')
		plt.savefig(os.path.join(out_dir, 'median_width_'+patient+'.png'), bbox_inches='tight')
		plt.close(fig)

	# dd = d['WS']
	# dd_global = []
	# a = dd['12-04-2017']
	# for k in dd.keys():
	# 	# print stats.shapiro(dd[k])
	# 	b = dd[k]
	# 	dd_global.append(b)
	# 	# print stats.kruskal(a, dd[k])
	# 	# print stats.ks_2samp(a, b) #Kolmogorov-Smirnov test
	# 	# print np.mean(a)/(np.mean(a)+np.mean(b))
	# # print stats.ttest_ind(a, b)
		

	## bootstrap procedure
	# N = 100
	# for n in xrange(1, N):






