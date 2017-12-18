#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse

from book_reader import *
import matplotlib.pyplot as py
import numpy as np
import scipy.signal as ss
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile
import pandas as pd
import sys
import os
import json

sys.path.append('/Users/magdalena/openbci/')

from obci.analysis.obci_signal_processing.tags import tags_file_writer as tags_writer
from obci.analysis.obci_signal_processing.tags import tag_utils

pages_db = './pages_db_single_channel.json'

channels_names = ['C3']


def filter_atoms(output_file, epoch_len, corrupted_epochs, ptspmV, atoms, fs, freq_range, width_range, width_coeff, amplitude_range):
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
                    if (width_range[0] <= width <= width_range[1]) and (amplitude_range[0] <= amplitude <= amplitude_range[1]) and (freq_range[0] <= frequency <= freq_range[1]):
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
    corrupted_epochs_energy = np.where(energy > energy_thr_upper)[0] 
    corrupted_epochs_diff = np.where(diff > diff_thr_upper)[0]
    return energy, diff, np.array(corrupted_epochs_energy)+1, np.array(corrupted_epochs_diff)+1

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

def get_rms_multiple_channels(book, selected_channels, channels_names, freq_spindle_range, window_width, percentile):
    signals = book.signals
    for i,ch in enumerate(selected_channels):
        idx = np.where(np.array(channels_names)==ch)
        channel = idx[0][0]+1
        energy, diff, corre, corrd = mark_corrupted_epochs(book, channel)
        to_remove = set(corre).difference(corrd)
        corrupted_epochs = [ep for ep in corre if ep not in to_remove]
        for booknumber in signals[channel]:
            if booknumber not in corrupted_epochs:
                try:
                    x = np.hstack((x, signals[channel][booknumber]))
                except NameError:
                    x = signals[channel][booknumber]
    d, c = ss.butter(2, np.array(freq_spindle_range) / (book.fs / 2), btype='bandpass')
    sig = ss.filtfilt(d, c, x)
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

def page_band_energy_from_reconstruction(output_file, channel, corrupted_epochs, atoms, freq_range):
    f = open(output_file, 'w')
    f.write('book_number, energy\n')
    chosen = []
    for booknumber in atoms[channel]:
        if booknumber not in corrupted_epochs:
            rec = b._reconstruct_page(booknumber, channel, freq_range)
            energy = np.sum(np.abs(rec)**2)
            f.write('{:d},{:.4f}\n'.format(booknumber, energy))
            chosen.append([booknumber, energy])
    f.close()
    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find activity waves for given bookfile. Bookfile is done by mp5 algorithm.')
    parser.add_argument('--name', required=True, help='Name of book file (without file extension)')
    args = parser.parse_args()
    f_name = args.name

    # f_name = "/Users/magdalena/projects/python/sleep_decompositions/part3/MB_20_12_2016_C3-ear_128.b"

    patient = os.path.basename(f_name).split('.')[0]
    out_dir = os.path.join('/home/mzieleniewska/sleep2/books/one_channel/', patient)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    print 'Working on patient: ', patient

    b = BookImporter(f_name)
    atoms = b.atoms
    signals = b.signals
    epoch_len = b.epoch_s
    fs = b.fs
    ptspmV = b.ptspmV
    number_of_epochs = len(signals[1])

    try:
        with open(pages_db) as f:
            pages_data = json.load(f)
        if f_name in pages_data.keys():
            pass
        else:
            pages_data[patient] = number_of_epochs
            with open(pages_db, 'w') as f_out:
                json.dump(pages_data, f_out)
    except IOError:
        with open(pages_db, 'w') as f:
            json.dump({patient: number_of_epochs}, f)

    signal = get_whole_signal(signals, ['C3'], epoch_len, number_of_epochs)

    window_width = 0.5
    freq_spindle_range = [10, 17]
    freq_delta_range = [0.5, 3]
    freq_theta_range = [3, 8]
    freq_alpha_range = [8, 12]
    freq_beta_range = [16, 25]

    rms_ss, rms_ampli_ss = get_rms_multiple_channels(b, ['C3'], channels_names, freq_spindle_range, window_width, 97.0)
    rms_delta, rms_ampli_delta = get_rms_multiple_channels(b, ['C3'], channels_names, freq_delta_range, 2, 97.0)
    rms_theta, rms_ampli_theta = get_rms_multiple_channels(b, ['C3'], channels_names, freq_theta_range, 1, 97.0)
    rms_alpha, rms_ampli_alpha = get_rms_multiple_channels(b, ['C3'], channels_names, freq_alpha_range, window_width, 97.0)
    rms_beta, rms_ampli_beta = get_rms_multiple_channels(b, ['C3'], channels_names, freq_beta_range, window_width, 97.0)

    # mark SS
    channel = 1
    energy, diff, corre, corrd = mark_corrupted_epochs(b, channel)
    to_remove = set(corre).difference(corrd)
    corrupted_epochs = [ep for ep in corre if ep not in to_remove]
    chann_atoms = atoms[channel]

    chosen_delta = filter_atoms(os.path.join(out_dir, patient + '_delta.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, freq_delta_range, [0.5, np.inf], 1.0, [rms_ampli_delta, np.inf])
    chosen_theta = filter_atoms(os.path.join(out_dir, patient + '_theta.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, freq_theta_range, [0.5, np.inf], 1.0, [rms_ampli_theta, np.inf])
    chosen_alpha = filter_atoms(os.path.join(out_dir, patient + '_alpha.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, freq_alpha_range, [1.5, np.inf], 1.0, [rms_ampli_alpha, np.inf])
    chosen_beta = filter_atoms(os.path.join(out_dir, patient + '_beta.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, freq_beta_range, [0.4, np.inf], 1.0, [rms_ampli_beta, np.inf])

    chosen_SS = filter_atoms(os.path.join(out_dir, patient + '_SS.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, freq_spindle_range, [0.4, np.inf], 1.0, [rms_ampli_ss, np.inf])
    
    recon = b._reconstruct_signal(channel, [0.5, 40])
    recon_spindle = b._reconstruct_signal(channel, freq_spindle_range)

    df = pd.read_csv(os.path.join(out_dir, patient + '_SS.csv'))
    offset = df['offset']
    width = df['width']
    amp = df['amplitude']
    ind_to_remove = []
    for i in xrange(len(offset)):
        frag = recon[int(offset[i]) : int(offset[i] + width[i] * fs)]
        frag_spindle = recon_spindle[int(offset[i]) : int(offset[i] + width[i] * fs)]
        if np.sum(np.abs(frag_spindle) ** 2) / np.sum(np.abs(frag) ** 2) < 0.2:
            ind_to_remove.append(i)
    df_new = df.drop(df.index[ind_to_remove])
    df_new.to_csv(os.path.join(out_dir, patient + '_SS.csv'))

    # chosen_beta = page_band_energy_from_reconstruction(os.path.join(out_dir, f_name + '_beta.csv'), channel, corrupted_epochs, atoms, freq_beta_range)
    # chosen_alpha = page_band_energy_from_reconstruction(os.path.join(out_dir, f_name + '_alpha.csv'), channel, corrupted_epochs, atoms, freq_alpha_range)
    # chosen_theta = page_band_energy_from_reconstruction(os.path.join(out_dir, f_name + '_theta.csv'), channel, corrupted_epochs, atoms, freq_theta_range)
    # chosen_delta = page_band_energy_from_reconstruction(os.path.join(out_dir, f_name + '_delta.csv'), channel, corrupted_epochs, atoms, freq_delta_range)

    ####### mark SWA
    recon = b._reconstruct_signal(channel, [0.5, 40])
    recon_swa = b._reconstruct_signal(channel, [0.5, 2])

    rms_ampli_swa = 7*np.sqrt(np.median(signal[0,:]**2)) 
    # print 'SWA amplitude 7 * median: ', rms_ampli_swa

    chosen_SWA = filter_atoms(os.path.join(out_dir, patient + '_SWA.csv'), epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [0.2,2], [0.5, 6], 1.0, [rms_ampli_swa, np.inf])

    df = pd.read_csv(os.path.join(out_dir, patient + '_SWA.csv'))
    offset = df['offset']
    width = df['width']
    amp = df['amplitude']
    ind_to_remove = []
    for i in xrange(len(offset)):
        frag = recon[int(offset[i]-0.5*fs) : int(offset[i]+width[i]*fs+0.5*fs)]
        frag_swa = recon_swa[int(offset[i]-0.5*fs) : int(offset[i]+width[i]*fs+0.5*fs)]
        if np.sum(np.abs(frag_swa)**2)/np.sum(np.abs(frag)**2) < 0.7:
            ind_to_remove.append(i)
    df_new = df.drop(df.index[ind_to_remove])
    df_new.to_csv(os.path.join(out_dir, patient + '_SWA.csv'))

