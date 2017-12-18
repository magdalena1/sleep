#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
License: For personnal, educationnal, and research purpose, this software is 
		provided under the Gnu GPL (V.3) license. To use this software in
		commercial application, please contact the authors.

Authors: Jaroslaw Zygierewicz (jarekz@fuw.edu.pl), 
		Piotr J. Durka (durka@fuw.edu.pl), Magdalena Zieleniewska (magdalena.zieleniewska@fuw.edu.pl)
Date   : September, 2014
'''

from __future__ import print_function, division
import numpy as np
import collections
import os


class BookImporter(object):
    def __init__(self, book_file):
        """
	Class for reading books from mp5 decomposition.

	Input:
		book_file   -- string -- book file
	"""

        super(BookImporter, self).__init__()

        try:
            f = open(book_file, 'rb')
        except IOError as e:
            print('IOError: ', os.strerror(e.errno))
            return
        version, data, signals, atoms, epoch_s = self._read_book(f)
        self.version = version[0]
        self.epoch_s = epoch_s
        self.atoms = atoms
        self.signals = signals
        self.fs = data[5]['Fs']
        self.ptspmV = data[5]['ptspmV']

    def _get_type(self, ident, f):
        if ident == 1:
            com_s = np.fromfile(f, '>u4', count=1)[0]
            if not com_s == 0:  ## comment
                return np.dtype([('comment', 'S' + str(com_s))])
            else:
                return None
        elif ident == 2:  ## header
            head_s = np.fromfile(f, '>u4', count=1)
            return None
        elif ident == 3:  ## www address
            www_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('www', 'S' + str(www_s))])
        elif ident == 4:  ## date
            date_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('date', 'S' + str(date_s))])
        elif ident == 5:  ## signal info
            sig_info_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('Fs', '>f4'), ('ptspmV', '>f4'),
                             ('chnl_cnt', '>u2')])
        elif ident == 6:  ## decomposition info
            dec_info_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('percent', '>f4'), ('maxiterations', '>u4'),
                             ('dict_size', '>u4'), ('dict_type', '>S1')])
        elif ident == 10:  # dirac
            # return
            atom_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('modulus', '>f4'), ('amplitude', '>f4'),
                             ('t', '>f4')])
        elif ident == 11:  # gauss
            atom_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('modulus', '>f4'), ('amplitude', '>f4'),
                             ('t', '>f4'), ('scale', '>f4')])
        elif ident == 12:  # sinus
            atom_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('modulus', '>f4'), ('amplitude', '>f4'),
                             ('f', '>f4'), ('phase', '>f4')])
        elif ident == 13:  # gabor
            atom_s = np.fromfile(f, '>u1', count=1)[0]
            return np.dtype([('modulus', '>f4'), ('amplitude', '>f4'),
                             ('t', '>f4'), ('scale', '>f4'),
                             ('f', '>f4'), ('phase', '>f4')])
        else:
            return None

    def _get_signal(self, f, epoch_nr, epoch_s):
        sig_s = np.fromfile(f, '>u4', count=1)[0]
        chnl_nr = np.fromfile(f, '>u2', count=1)[0]
        signal = np.fromfile(f, '>f4', count=epoch_s)
        return chnl_nr, signal

    def _get_atoms(self, f):
        atoms = list()
        atoms_s = np.fromfile(f, '>u4', count=1)[0]
        end_pos = f.tell() + atoms_s
        a_chnl_nr = np.fromfile(f, '>u2', count=1)[0]
        while f.tell() < end_pos:
            ident = np.fromfile(f, '>u1', count=1)
            atom = np.fromfile(f, self._get_type(ident[0], f), count=1)[0]
            atoms.append({'params': atom, 'type': ident[0]})
        return atoms, a_chnl_nr

    def _read_book(self, f):
        version = np.fromfile(f, 'S6', count=1)
        data = {}
        ident = np.fromfile(f, 'u1', count=1)[0]
        ct = self._get_type(ident, f)
        signals = collections.defaultdict(dict)
        atoms = collections.defaultdict(dict)
        while ident:
            if ct:
                point = np.fromfile(f, ct, count=1)[0]
                data[ident] = point
            elif ident == 7:
                data_s = np.fromfile(f, '>u4', count=1)[0]
                epoch_nr = np.fromfile(f, '>u2', count=1)[0]
                epoch_s = np.fromfile(f, '>u4', count=1)[0]
            elif ident == 8:
                chnl_nr, signal = self._get_signal(f, epoch_nr, epoch_s)
                signals[chnl_nr][epoch_nr] = signal
            elif ident == 9:
                pl = f.tell()
                atom, a_chnl_nr = self._get_atoms(f)
                atoms[a_chnl_nr][epoch_nr] = atom
            ident = np.fromfile(f, '>u1', count=1)
            if ident:
                ident = ident[0]
            ct = self._get_type(ident, f)
        return version, data, signals, atoms, epoch_s

    def _gabor(self, amplitude, position, scale, afrequency, phase):
        time = np.linspace(0, self.epoch_s/self.fs, self.epoch_s)
        width = scale
        frequency = afrequency*2*np.pi
        signal = amplitude*np.exp(-np.pi*((time-position)/width)**2)*np.cos(frequency*(time-position) + phase)
        return signal

    def _sinus(self, amplitude, frequency, phase):
        time = np.linspace(0, self.epoch_s/self.fs, self.epoch_s)
        frequency = frequency*2*np.pi
        recon = amplitude*np.cos(frequency*time + phase)
        return recon

    def _reconstruct_signal(self, channel, freq_range):
        channel_atoms = self.atoms[channel]
        reconstruction = np.zeros(len(channel_atoms)*self.epoch_s)
        for i,booknumber in enumerate(channel_atoms):
            recon_page = np.zeros(self.epoch_s)
            for it,atom in enumerate(channel_atoms[booknumber]):
                if atom['type'] in [12, 13]:
                    frequency = atom['params']['f']*self.fs/2
                    amplitude = atom['params']['amplitude']/self.ptspmV
                    phase 	  = atom['params']['phase']
                    if (freq_range[0] < frequency < freq_range[1]):
                        if atom['type'] == 13:
                            position  = atom['params']['t']/self.fs
                            width     = atom['params']['scale']/self.fs
                            recon_page = recon_page + self._gabor(amplitude, position, width, frequency, phase)
                        else:
                            recon_page = recon_page + self._sinus(amplitude, frequency, phase)
            reconstruction[i*self.epoch_s:(i+1)*self.epoch_s] = recon_page
        return reconstruction

    def _reconstruct_page(self, booknumber, channel, freq_range):
        channel_atoms = self.atoms[channel]
        recon_page = np.zeros(self.epoch_s)
        for it,atom in enumerate(channel_atoms[booknumber]):
            if atom['type'] in [12, 13]:
                frequency = atom['params']['f']*self.fs/2
                amplitude = atom['params']['amplitude']/self.ptspmV
                phase     = atom['params']['phase']
                if (freq_range[0] < frequency < freq_range[1]):
                    if atom['type'] == 13:
                        position  = atom['params']['t']/self.fs
                        width     = atom['params']['scale']/self.fs
                        recon_page = recon_page + self._gabor(amplitude, position, width, frequency, phase)
                    else:
                        recon_page = recon_page + self._sinus(amplitude, frequency, phase)
        return recon_page


