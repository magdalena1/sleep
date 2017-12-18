#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import os
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import nolds
import seaborn as sns
import operator


from statsmodels.sandbox.stats.multicomp import multipletests as mt
from scipy.optimize import curve_fit, minimize


WAVE_TYPES = ['spindle', 'SWA']

OUT_DIR = '/home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms/params_results/'


def plot_parameter_for_groups(df, parameter):
	fig, axs = plt.subplots(len(WAVE_TYPES), 1, figsize=(10, 7))
	groups = np.unique(df.crs_group)
	x = np.array(range(0, len(groups)))
	for (i, wave_type) in enumerate(WAVE_TYPES):
		key = parameter+'_'+wave_type
		y = [np.nanmean(df[key][df.crs_group==score]) for score in groups]
		y_std = [np.nanstd(df[key][df.crs_group==score]) for score in groups]
		print y
		# axs[i].boxplot(y)
		# slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) #linear regression
		# axs[i].plot(x, intercept + slope*x, 'r')
		axs[i].errorbar(x, y, yerr=y_std, fmt='o')
		# axs[i].text(0.85, 0.1, 'p = ' +  str(np.round(p_value, 3)), transform=axs[i].transAxes)
		axs[i].set_xlim([-0.25, len(x)-0.75])
		axs[i].set_ylabel(key)
		axs[i].set_xticks(x)
		axs[i].set_title(wave_type)
		axs[i].set_xticklabels(groups)
	return fig


def spectrum_model(f, alpha, b):
	return b / (f ** alpha)


def spectrum_model2(f, alpha, b):
	return b * (f ** alpha)


def func_for_datapoints(w, p):
	return lambda v: np.sum((v[1]*w**v[0] - p)**2)


def fit_spectrum_model(spectrum, frequencies):
	popt, pcov = curve_fit(spectrum_model, frequencies, spectrum, maxfev=1000)
	s_normalised = spectrum-spectrum_model(frequencies, *popt)
	return s_normalised, popt


def fit_spectrum_model2(spectrum, frequencies):
	constraints = []
	for p, w in zip(spectrum, frequencies):
		constraints.append({'type': 'ineq', 'fun': lambda v: p - v[1]*w**v[0]})
	res = minimize(func_for_datapoints(frequencies, spectrum), [-1, 1], method='COBYLA')
	return res

def plot_waterfall(df):
	df_sorted = df.sort_values(by='crs_group')
	df_sorted = df_sorted.reset_index(drop=True)

	spectra = []

	for rec in df_sorted.rec_id:
		spectrum = np.load(os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/patients_spectra_Cz_05_25Hz/', rec+'_ears_full_128_power.npy'))
		frequencies = np.load(os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/patients_spectra_Cz_05_25Hz/', rec+'_ears_full_128_freq.npy'))
		spectra.append(spectrum)

	ids_EMCS = [min(df_sorted.index[df_sorted.crs_group=='EMCS'].tolist()), max(df_sorted.index[df_sorted.crs_group=='EMCS'].tolist())]
	ids_MCS = [min(df_sorted.index[df_sorted.crs_group=='MCS'].tolist()), max(df_sorted.index[df_sorted.crs_group=='MCS'].tolist())]
	ids_VS = [min(df_sorted.index[df_sorted.crs_group=='VS'].tolist()), max(df_sorted.index[df_sorted.crs_group=='VS'].tolist())]
	ids = [ids_EMCS, ids_MCS, ids_VS]

	spectra = np.array(spectra)
	fig = plt.figure(facecolor='k')
	ax = fig.add_subplot(111, axisbg='k')
	ny = spectra.shape[0]
	for iy in range(ny):
		if (iy >= ids_EMCS[0]) and (iy <= ids_EMCS[1]):
			curve_col = 'm'
		elif (iy >= ids_MCS[0]) and (iy <= ids_MCS[1]):
			curve_col = 'g'
		elif (iy >= ids_VS[0]) and (iy <= ids_VS[1]):
			curve_col = 'y'
		offset = (ny-iy)*0.01
		ax.plot(frequencies, spectra[iy]+offset, curve_col, zorder=(iy+1)*2)
		ax.fill_between(frequencies, spectra[iy]+offset, offset, facecolor='k', lw=0, zorder=(iy+1)*2-1)
	ax.set_xlim([0.5, 20])
	plt.savefig(os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/patients_spectra_Cz_05_25Hz/waterplot_spectra_sorted.png'))
	plt.show()


def plot_swarmplot_for_params(df):
	sns.set_context("poster", font_scale=1.5)
	sns.axes_style("whitegrid")
	a = ["#e6194b","#3cb44b","#ffe119","#0082c8","#f58231","#911eb4","#46f0f0","#f032e6","#d2f53c","#008080", 
		 "#342D7E","#aa6e28","#254117","#800000","#64E986","#808000","#EE9A4D","#000080","#808080","#000000", 
		 "#8A4117"]
	for key_id in xrange(8, len(df.columns)):
		param_name = df.columns[key_id]
		fig = plt.figure(figsize=(40, 22))
		ax = fig.add_subplot(111)
		# ax.set_yscale("log")
		plt.title(param_name)
		sns.swarmplot(x='crs_group', y=param_name, data=df, hue='patient_id', palette=a, size=18)
		L = plt.legend(prop={'size': 18}, loc='center left', bbox_to_anchor=(0.92, 0.5))
		fig.savefig(os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/', 'patients_results', 'swarmplot_params_new_reader', "pdf_" + param_name + ".pdf"), format='pdf')
		plt.close(fig)


if __name__ == '__main__':

	crs_file = '/home/mzieleniewska/sleep2/books/crs.csv'
	df = pd.read_csv(crs_file)

	df['rec_id'] = df.patient_id + '_' + df.recording_date.str.split('-').apply(lambda parts: "_".join(parts))

	df = df.drop(df.index[df["rec_id"] == "AS_19_11_2016"], inplace=False).reset_index(drop=True)
	# df.drop(df.index[df["patient_id"] == "MB"], inplace=True)

	conv_bd = [datetime.strptime(d, '%d-%m-%Y') for d in df.birth_date]
	conv_rd = [datetime.strptime(d, '%d-%m-%Y') for d in df.recording_date]
	df['age'] = [relativedelta(conv_rd[i], conv_bd[i]).years for i in xrange(len(conv_bd))]
	df['crs_group'] = df.crs_score
	df.crs_group.loc[(df.crs_group=='MCS-') | (df.crs_group=='MCS+')] = 'MCS'

	# #get profile parameters
	# f_list = glob.glob('/home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/params/*_params.csv')

	# ##remove certain files
	# matching = [s for s in f_list if "AS_19_11_2016" in s]
	# if len(matching) == 1:
	# 	f_list.remove(matching[0])
	# else:
	# 	[f_list.remove(m) for m in matching]

	df_param = pd.DataFrame()
	for rec_id in df["rec_id"]:
		f = os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/params/', rec_id + '_params.csv')
		df_temp = pd.read_csv(f, index_col=0)
		if df_param.empty:
			df_param = pd.DataFrame(columns=df_temp.columns)
		name = os.path.basename(f).split('.')[0].split('_')
		rec_id = '_'.join(name[:-2])
		if df.rec_id.str.contains(rec_id).any():
			df_param.loc[df.index[df.rec_id.str.contains(rec_id)][0]] = df_temp.loc[0].tolist()
	df_param = df_param.sort_index()
	df = pd.concat([df, df_param], axis=1)

	# plot_waterfall(df)

	patients_ids = np.unique(df.patient_id)
	patients_start_age = []
	for p in patients_ids:
		print p, np.min(df.loc[df.index[df.patient_id.str.contains(p)]].age)
		patients_start_age.append(np.min(df.loc[df.index[df.patient_id.str.contains(p)]].age))
	print "age: ", np.mean(patients_start_age), "+/-", np.std(patients_start_age) 
	
	groups = np.unique(df.crs_group)
	p_vals = dict()
	for key_id in xrange(8, len(df.columns)):
		key = df.columns[key_id]
		y = [df[key][df.crs_group==score] for score in groups]
		# print stats.ttest_ind(y[0].dropna(), y[2].dropna())
		s = stats.mannwhitneyu(y[1].dropna(), y[2].dropna())
		p_vals[key] = s[1]
		print key, np.mean(y[1]), np.mean(y[2]), ": p-value =", s[1]

	p_fdr = mt(p_vals.values(), alpha=0.05, method='fdr_bh')
	p_vals_fdr = dict()
	for k, key in enumerate(p_vals.keys()):
		p_vals_fdr[key] = p_fdr[1][k]

	#sorted_p = OrderedDict(sorted(p_vals_fdr.items(), key=lambda(k,v):(v,k)))
	sorted_p = sorted(p_vals_fdr.items(), key=operator.itemgetter(1))
	selected_p = [(param, '%.5f' % p_value) for (param, p_value) in sorted_p]# if p_value < 0.05]
	print selected_p
	param_names = [param for (param, p_value) in selected_p]
	# param_names = param_names[:7]
	param_names.append('crs_group')

	# df.replace('', np.nan, inplace=True)
	# df.drop(["frequency_mse_spindle"], axis=1, inplace=True)
	df.to_csv('/home/mzieleniewska/empi/from_hpc/data/smp/classification_parameters.csv')

	df_pairplot = df[['crs_group', 'power_spindle', 'profile_dfa_spindle', 'calinski_harabaz_spindle']]
	fig = sns.pairplot(df_pairplot, hue='crs_group', palette="husl", size=15, plot_kws={"s": 40})
	fig.savefig('/home/mzieleniewska/empi/from_hpc/data/smp/pairplot_ss.pdf', format='pdf')
	
	df_pairplot = df[['crs_group', 'power_SWA', 'profile_dfa_SWA', 'calinski_harabaz_SWA']]
	fig = sns.pairplot(df_pairplot, hue='crs_group', palette="husl", size=15, plot_kws={"s": 40})
	# plt.tight_layout()
	fig.savefig('/home/mzieleniewska/empi/from_hpc/data/smp/pairplot_swa.pdf', format='pdf')

	plot_swarmplot_for_params(df)



	# for t in L.get_texts():
	# 	t.set_text(l_labels[t.get_text()])
	# plt.axhline(y=0.5, linewidth=1.5, color='k')
	# fig.savefig(os.path.join(WORKFOLDER, "fig_AUC_max.png"))


	#dfp_vm = dfp[(dfp["crs_group"]=="VS") | (dfp["crs_group"]=="MCS")]
	#TODO: iterate over pairs of variables
	#sns.pairplot(data=dfp_vm, hue="crs_group", markers=["o", "x"], palette="husl", vars=['std_density_SWA','spectral_entropy'])


	# grps = pd.unique(df.crs_group.values)
	# for key_id in xrange(8, len(df.columns)):
	# 	key = df.columns[key_id]
	# 	d_data = {grp: df[key][df.crs_group==grp] for grp in grps}
	# 	F, p = stats.f_oneway(d_data['EMCS'].dropna(), d_data['MCS'].dropna(), d_data['VS'].dropna())
	# 	print key, p
		# df.boxplot(key, by='crs_group', figsize=(12, 8))
		# plt.savefig(os.path.join(OUT_DIR, key+'_boxplot.png'))










