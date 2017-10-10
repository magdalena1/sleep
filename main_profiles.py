#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import json
import glob

from create_histograms import create_histograms
from save_structures_params import save_structures_params
from create_profiles import create_profiles


if __name__ == "__main__":

	book_dir = '/home/mzieleniewska/sleep2/books/full_cap'

	file_list = glob.glob(os.path.join(book_dir, "*.b"))

	for f in file_list:
		name = os.path.basename(f).split('.')[0]
		directory = os.path.dirname(f)
		create_profiles(name, directory)
		create_histograms(name, os.path.join(directory, name), 3)
		save_structures_params(name, os.path.join(directory, name))

