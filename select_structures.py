#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os.path
import sys
import pandas as pd
import argparse


DEFAULT_THRESHOLD = 12.


def main():
	parser = argparse.ArgumentParser(description='Selects transients based on amplitude criterium.')
	parser.add_argument('files', nargs='+', metavar='file', help='path to *occ.csv files')
	parser.add_argument('-t', dest='threshold', default=DEFAULT_THRESHOLD, 
						help='amplitude threshold for Gabor selection (default: '+str(DEFAULT_THRESHOLD)+')')
	namespace = parser.parse_args()

	for path_to_csv in namespace.files:
		directory = os.path.dirname(path_to_csv)
		file_name = os.path.basename(path_to_csv)
		name = "_".join(file_name.split("_")[:3])
		df = pd.read_csv(path_to_csv, index_col=0)
		df = df.drop(df[df.amplitude < float(namespace.threshold)].index, inplace=False).reset_index(drop=True)
		df.to_csv(os.path.join(directory, name + "_occ_sel.csv"))


if __name__ == '__main__':
	main()
	sys.exit(0)

			

			

			

