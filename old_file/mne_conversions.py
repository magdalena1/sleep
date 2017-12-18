#coding: utf8
import os.path
import mne
import numpy as np
from obci.analysis.obci_signal_processing.read_manager import ReadManager
from obci.analysis.obci_signal_processing.signal.read_data_source import MemoryDataSource
from obci.analysis.obci_signal_processing.signal.read_info_source import MemoryInfoSource

def mne_to_rm_list(e_mne):
    epochs_l = []
    for n, s in enumerate(sorted(e_mne.event_id.values())):
        mask = e_mne.events[:,2] == s
        epochs = e_mne[mask].get_data()*1e6
        epochs_type = []
        for epoch in epochs:
            mds = MemoryDataSource(epoch)
            mis = MemoryInfoSource(p_params={'sampling_frequency':str(e_mne.info['sfreq']),
                                             'channels_names': e_mne.ch_names,
                                             'number_of_channels': len(e_mne.ch_names),
                                             'file':e_mne.info['filename']

                                             })
            rm = ReadManager(p_data_source=mds, p_info_source=mis, p_tags_source=None)
            epochs_type.append(rm)
        epochs_l.append(epochs_type)
    return epochs_l


def mne_info_from_rm(rm):
    chnames = rm.get_param('channels_names')
    sfreq = float(rm.get_param('sampling_frequency'))
    ch_types = [chtype(i) for i in chnames]
    info = mne.create_info(ch_names=chnames, sfreq=sfreq, ch_types=ch_types, montage='standard_1005')
    info['filename'] = os.path.basename(rm.get_param('file'))
    return info




def read_manager_to_mne(epochs, baseline=None):
    '''Returns all epochs in one mne.EpochsArray object and slices for every tagtype'''
    all_epochs = []
    tag_type_slices = []
    last_ep_nr = 0
    for tagtype_e in epochs:
        for epoch in tagtype_e:
            all_epochs.append(epoch.get_samples()*1e-6)
        len_tagtype = len(tagtype_e)
        tag_type_slices.append(slice(last_ep_nr, len_tagtype + last_ep_nr))
        last_ep_nr = len_tagtype + last_ep_nr
    info = mne_info_from_rm(epoch)

    min_length = min(i.shape[1] for i in all_epochs)

    all_epochs = [i[:,0:min_length] for i in all_epochs]
    all_epochs_np = np.stack(all_epochs)

    event_types = np.ones((len(all_epochs), 3), dtype = int)
    event_types[:,0] = np.arange(0, len(all_epochs), 1)
    for n, s in enumerate(tag_type_slices):
        event_types[s,2] = n

    print info
    e_mne = mne.EpochsArray(all_epochs_np, info, tmin=baseline, baseline=(baseline,0), events=event_types)
    return e_mne, tag_type_slices

def read_manager_continious_to_mne(rm):
    info = mne_info_from_rm(rm)
    raw = mne.io.RawArray(rm.get_samples()*1e-6, info)
    return raw


def chtype(name):
    ineeg = [u'Fp1',
             u'Fpz',
             u'Fp2',
             u'F7',
             u'F3',
             u'Fz',
             u'F4',
             u'F8',
             u'M1',
             u'T3',
             u'C3',
             u'Cz',
             u'C4',
             u'T4',
             u'M2',
             u'T5',
             u'P3',
             u'Pz',
             u'P4',
             u'T6',
             u'O1',
             u'Oz',
             u'O2',]
    if name in ineeg:
        return 'eeg'
    elif name.lower() == 'eog':
        return 'eog'
    else:
        return 'misc'
