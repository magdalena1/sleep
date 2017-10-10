#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pylab as py
import pandas as pd
import numpy as np
import json
import IPython

WAVE_TYPES = ['SS', 'SWA', 'delta', 'theta', 'alpha', 'beta'] 

pages_db = './pages_db_single_channel.json'

fs = 128.0

def create_histograms(name, directory, orig_bin_width):
    data_name = "{}/{}".format(directory, name) + "_{}.csv"
    hipnogram_out_name = "{}/{}".format(directory, name) + "_{}_hipnogram.csv"
    #data_name = name + "_{}.csv"
    #hipnogram_out_name = name + "_{}_hipnogram.csv"

    with open(pages_db) as f:
        # parts = name.split('_')
        # key = '_'.join(parts[:-1])
        key_temp = name.split('/')[-1]
        key = key_temp.split('.')[0]
        print key
        pages_data = json.load(f)
        total_len = float(pages_data[key]*20)/3600

    for (i, wave_type) in enumerate(WAVE_TYPES):
        data = pd.read_csv(data_name.format(wave_type))
        # Absolute position is given in hours.
        data['absolute_position'] = ((data['book_number']-1)*20. + data['position'])/3600
        if wave_type == 'SWA':
            bin_width = 0.33#1./3
        else:
            bin_width = orig_bin_width

        number_of_bins = int(total_len*60/bin_width+1)
        bins = np.zeros(number_of_bins)
        time = np.linspace(0, float(number_of_bins*bin_width)/60, len(bins))

        # IPython.embed()
        # time = np.arange(0, total_len*60, bin_width)/60
        for (index, row) in data.iterrows():
            if wave_type == 'SWA':
                bins[int(np.floor(row['absolute_position']*60/bin_width))] += row['struct_len']/fs
            elif wave_type == 'SS':
                bins[int(np.floor(row['absolute_position']*60/bin_width))] += 1
            else:
                bins[int(np.floor(row['absolute_position']*60/bin_width))] += 2*row['amplitude'] #row['energy']
        if wave_type == 'SWA':
            bins = (bins/20)*100
            bins[np.where(bins>100)[0]] = 100
        # elif wave_type in ['theta', 'alpha', 'beta']:
        #     bins = bins/(orig_bin_width*60)
        sig = pd.DataFrame({'time': time, 'occurences': bins})
        sig.to_csv(hipnogram_out_name.format(wave_type))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create histrograms for profiles')
    parser.add_argument('--name', required=True,
                    help='Base name of profiles (without suffix eg. "_alpha.csv").')
    parser.add_argument('--dir', required=True, help='Path to directory')
    parser.add_argument('--bin-width', type=int, required=True, help='Bin width given in minutes')
    args = parser.parse_args()

    # name='/Users/magdalena/projects/python/sleep_decompositions/part3/AC_C4-RM_128.b'
    # dir='/Users/magdalena/projects/python/sleep_decompositions/part3/'
    # bin_width=3
    # create_histograms(name, dir, bin_width)

    create_histograms(args.name, args.dir, args.bin_width)
