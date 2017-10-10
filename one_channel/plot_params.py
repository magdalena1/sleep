#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib.pylab as py
import pandas as pd
import numpy as np

WAVE_TYPES = ['SS', 'SWA']

if __name__ == '__main__':

	out_dir = '/home/mzieleniewska/sleep2/results/distributions'
	book_dir = '/home/mzieleniewska/sleep2/books/one_channel'

	file_list = glob.glob(os.path.join(book_dir, "*.b"))
	#.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))