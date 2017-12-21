#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import argparse

from statsmodels.sandbox.stats.multicomp import multipletests as mt
from scipy.optimize import curve_fit, minimize


def main():
	parser = argparse.ArgumentParser(description='Group parameters.')
	parser.add_argument('files', nargs='+', metavar='file', help='path to *params.csv files')
	namespace = parser.parse_args()

	df_param = pd.DataFrame()
	for path_to_csv in namespace.files:
		name = os.path.basename(path_to_csv)
		df_temp = pd.read_csv(path_to_csv, index_col=0)
		df_temp.insert(loc=0, column="patient_id", value=name.split("_")[0])
		df_param = df_param.append([df_temp])
	df_param.to_csv(os.path.join("/".join(os.path.dirname(path_to_csv).split("/")[:-1]), 'classification_parameters.csv'))


if __name__ == '__main__':
	main()


	# df_param = pd.DataFrame()
	# for rec_id in df["rec_id"]:
	# 	f = os.path.join('/home/mzieleniewska/empi/from_hpc/data/smp/control_99rms_new_reader/params/', rec_id + '_params.csv')
	# 	df_temp = pd.read_csv(f, index_col=0)
	# 	if df_param.empty:
	# 		df_param = pd.DataFrame(columns=df_temp.columns)
	# 	name = os.path.basename(f).split('.')[0].split('_')
	# 	rec_id = '_'.join(name[:-2])
	# 	if df.rec_id.str.contains(rec_id).any():
	# 		df_param.loc[df.index[df.rec_id.str.contains(rec_id)][0]] = df_temp.loc[0].tolist()
	# df_param = df_param.sort_index()
	# df = pd.concat([df, df_param], axis=1)

	# df.to_csv('/home/mzieleniewska/empi/from_hpc/data/smp/classification_parameters.csv')

