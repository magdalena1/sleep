#!/usr/bin/env python
# -*- coding: utf-8 -*-
# based on obci.analysis.p300.analysis_offline
# Marian Dovgialo
import os
from scipy import signal, stats
import numpy as np
import pylab as pb

from copy import deepcopy
import json

from obci.analysis.obci_signal_processing import read_manager
from obci.analysis.obci_signal_processing.signal import read_info_source, read_data_source
from obci.analysis.obci_signal_processing.tags import read_tags_source
from obci.analysis.obci_signal_processing.tags.smart_tag_definition import SmartTagDurationDefinition
from obci.analysis.obci_signal_processing.tags.tags_file_writer import TagsFileWriter
from obci.analysis.obci_signal_processing.tags.tags_file_reader import TagsFileReader
from obci.analysis.obci_signal_processing.smart_tags_manager import SmartTagsManager


from mne_conversions import read_manager_continious_to_mne, read_manager_to_mne, mne_to_rm_list
import mne

import autoreject
import seaborn as sns

OUTDIR = './'

from collections import namedtuple
Pos = namedtuple('Pos', ['x', 'y'])
map1020 = {'eog': Pos(0,0), 'Fp1': Pos(1,0), 'Fpz': Pos(2,0), 'Fp2': Pos(3,0),  'Null': Pos(4,0),
            'F7': Pos(0,1),  'F3': Pos(1,1),  'Fz': Pos(2,1),  'F4': Pos(3,1),  'F8': Pos(4,1),
            'T3': Pos(0,2),  'C3': Pos(1,2),  'Cz': Pos(2,2),  'C4': Pos(3,2),  'T4': Pos(4,2),
            'T5': Pos(0,3),  'P3': Pos(1,3),  'Pz': Pos(2,3),  'P4': Pos(3,3),  'T6': Pos(4,3),
            'M1': Pos(0,4),  'O1': Pos(1,4),  'Oz': Pos(2,4),  'O2': Pos(3,4),  'M2': Pos(4,4),
                    }


def get_filelist(filename, args):
    if os.path.exists(filename) and not args:
        print 'pracuję nad pojedyńczym plikiem', filename
        filelist = [filename,]
    elif args:
        print 'pracuję nad listą plików', filename, args
        filelist = [filename]+list(args)
        filename = '_'.join(os.path.basename(filename).split('_')[0:-1])#+'P300cz_latest'
    else:
        print 'pracuję ze wzorem', filename
        filelist = glob.glob(filename)
    filelist = [fname for fname in filelist if "obci.raw" in fname and not "etr" in fname]
    return filelist, filelist[0]

def get_tagfilters(blok_type):
    if not blok_type: return [None,]

    if blok_type==1:
        def target_func(tag):
            try: return tag['desc']['blok_type']=='1' and tag['desc']['type']=='target'
            except: return False
        def nontarget_func(tag):
            try: return tag['desc']['blok_type']=='1' and tag['desc']['type']=='nontarget'
            except: return False

    elif blok_type==2:
        def target_func(tag):
            try: return tag['desc']['blok_type']=='2' and tag['desc']['type']=='target'
            except: return False
        def nontarget_func(tag):
            try: return tag['desc']['blok_type']=='2' and tag['desc']['type']=='nontarget'
            except: return False

    elif blok_type=='both':
        def target_func(tag):
            try: return tag['desc']['type']=='target'
            except: return False
        def nontarget_func(tag):
            try: return tag['desc']['type']=='nontarget'
            except: return False

    elif blok_type=='local':
        def target_func(tag):
            try: return tag['desc']['type_local']=='dewiant'
            except: return False
        def nontarget_func(tag):
            try: return tag['desc']['type_local']=='standard'
            except: return False

    elif blok_type=='global':
        def target_func(tag):
            try: return tag['desc']['type_global']=='target'
            except: return False
        def nontarget_func(tag):
            try: return tag['desc']['type_global']=='nontarget'
            except: return False

    return target_func, nontarget_func




def savetags(stags, filename, start_offset=0, duration=0.1):
    '''Create tags XML from smart tag list'''
    writer = TagsFileWriter(filename)
    for stag in stags:
        tag = stag.get_tags()[0]
        tag['start_timestamp'] += start_offset
        tag['end_timestamp'] += duration + start_offset
        writer.tag_received(tag)
    writer.finish_saving(0.0)

def get_microvolt_samples(stag,channel=None):
    '''Does get_samples on smart tag (read manager), but multiplied by channel gain'''
    if not channel: #returns for all channels
        gains = np.array([float(i) for i in stag.get_param('channels_gains')], ndmin=2).T
        return stag.get_samples()*gains
    else: #returns for specific channel
        chN=stag.get_param('channels_names').index(channel)
        gain = stag.get_param('channels_gains')[chN]
        return stag.get_channel_samples(channel)*float(gain)


def stag_amp_grad_art_rej(stags, amp_thr = 200, grad_thr = 50):
    '''Do artifact rejection on smart tags,
    returns clean tags and dirty tags,
    rejected by amplitude threshold and difference between samples treshold
    amplutude is provided in microVolts'''
    clean_stags = []
    dirty_stags = []

    for stag in stags:
        samples = stag.get_samples()
        #~ print 'DIRTY?', np.max(np.abs(samples))
        diff = np.diff(samples, axis=0)
        if amp_thr>np.max(np.abs(samples)) and grad_thr>np.max(np.abs(diff)):
            clean_stags.append(stag)
        else:
            dirty_stags.append(stag)
    return clean_stags, dirty_stags

def align_tags(rm, tag_correction_chnls, start_offset=-0.1, duration=0.3, thr=None, reverse=False, offset=0):
    '''aligns tags in read manager to start of sudden change std => 3 in either tag_correction_chnls list
    searches for that in window [start_offset+tag_time; tag_time+duration]
    if no such change occures - does nothing to the tag - reverse - searches for end of stimulation
    offset - offset in seconds to add forcibly
    '''
    tags = rm.get_tags()
    Fs = float(rm.get_param('sampling_frequency'))
    trigger_chnl=np.zeros(int(rm.get_param('number_of_samples')))
    for tag_correction_chnl in tag_correction_chnls:
        trigger_chnl += np.abs(rm.get_channel_samples(tag_correction_chnl))
    if not thr:
        thr=3*np.std(trigger_chnl)+np.mean(trigger_chnl)
        maksimum = trigger_chnl.max()
        if thr>0.5*maksimum: thr = 0.5*maksimum

    for tag in tags:
        start = int((tag['start_timestamp']+start_offset)*Fs)
        end = int((tag['start_timestamp']+start_offset+duration)*Fs)
        try:
            if reverse:
                trig_pos_s_r = np.argmax(np.flipud(trigger_chnl[start:end]>thr))
                trig_pos_s = (end-start-1)-trig_pos_s_r
            else:
                trig_pos_s = np.argmax(trigger_chnl[start:end]>thr) #will find first True, or first False if no Trues
        except ValueError:
            #~ print e
            tag['start_timestamp']+=offset
            tag['end_timestamp']+=offset
            continue
        #Debuging code:
##        print trig_pos_s, Fs, reverse,
##        print 'thr', thr, 'value at pos', trigger_chnl[start+trig_pos_s], trigger_chnl[start+trig_pos_s]>thr
##        pb.plot(np.linspace(0, (end-start)/Fs, len(trigger_chnl[start:end])), trigger_chnl[start:end])
##        pb.axvline(trig_pos_s/Fs, color='k')
##        pb.title(str(tag))
##        pb.show()
        #Debug code end
        if trigger_chnl[start+trig_pos_s]>thr:
            trig_pos_t = trig_pos_s*1.0/Fs
            tag_change = trig_pos_t+start_offset
            #~ print 'TAG DIFF', tag_change
            tag['start_timestamp']+=tag_change
            tag['end_timestamp']+=tag_change
        tag['start_timestamp']+=offset
        tag['end_timestamp']+=offset
    rm.set_tags(tags)

def fit_eog_ica(rm, eog_chnl='eog',
                   montage=None, ds='',
                   use_eog_events=False, manual=False,
                   rejection_dict = dict(eeg=0.000150,
                                         eog=0.000250  # V
                                         ),
                   correlation_treshhold = 0.25,
                                         ):
    """
    rm - read manager with training data for ICA
    eog_chnl - channel to use as EOG source
    montage - montage of the read manager (for logging and filenames of generated images)
    use_eog_events True - split to EOG epochs - do ICA
    use eog events False - use whole file
    use_eog_events None - only create eog events
    manual True/False - shows ICA components, ICA components map
        prints correlations with EOG and then lets user write space seperated
        indexes of components to remove

    Returns
        - fitted mne.ICA object to be used in remove_eog_ica to correct read manager
        - list of bad components
        - detected eog events 
    """
    print('removing eog artifact')
    raw = read_manager_continious_to_mne(rm)
    n = len(raw.ch_names)
    print('n chnls {}'.format(n))
    raw.plot(block=True, show=True, scalings='auto', title='Simple preview of signal',n_channels=n)
    events = mne.preprocessing.find_eog_events(raw, ch_name=eog_chnl)
    print('EOG EVENTS\n', events)
    if use_eog_events is None:
        return rm, events
    ica = mne.preprocessing.ICA(method='extended-infomax', max_pca_components=None, n_components = None)
    #ica = mne.preprocessing.ICA()

    if use_eog_events == True:
        eog = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_chnl)
        ica.fit(eog, reject=rejection_dict,
                picks=mne.pick_types(raw.info, eeg=True, eog=True),
                )
    elif use_eog_events == False:
        ica.fit(raw, reject=rejection_dict, tstep=0.3)
    bads_, scores = ica.find_bads_eog(raw, ch_name=eog_chnl, threshold=1.5)

    bads = []
    for nr_c, score in enumerate(scores):
        if np.abs(score)>correlation_treshhold:
            bads.append(nr_c)

    print(ica)

    filename = 'ICA_eog_' + os.path.basename(ds) + '_{}'.format(montage)

    log = open(os.path.join(OUTDIR,"ica_maps", filename+'.txt'), 'w')
    print('CORRELATION SCORES:')
    log.write('CORRELATION SCORES:\n')
    for nr, score in enumerate(scores):
        msg = '{} {} {}'.format(nr, score, '*' if np.abs(score)>correlation_treshhold else '')
        print msg
        log.write(msg+'\n')
    log.close()
    title = 'Components with artifacts: {}, Correlation {} > {}'.format(bads, eog_chnl, correlation_treshhold)
    fig = ica.plot_components(res=128, show=False, title=title, colorbar = True)
    fig_2 = ica.plot_sources(raw)
    print u'Wybrane złe komponenty:', bads
    if manual:

        print('Wpisz indeksy komponent rozdzielone spacjami jeśli chcesz nadpisać (po zamknięciu okienek)\n'
              'jeśli nic nie wpiszesz użyją się wybrane automatycznie [potwierdź ENTERem]\n'
              'jeśli nie chcesz usuwać żadnej wpisz -1.')
        pb.show()
        good = False
        while not good:
            try:
                inp = raw_input()
                if inp.split():
                    man_bads = [int(i) for i in inp.split() if int(i)>=0 and int(i)<n]
                    if not man_bads and not int(inp)==-1: raise Exception
                    bads=man_bads
                good = True
            except Exception:
                print('Błąd, wpisz jeszcze raz\n')
        print u'Wybrane złe komponenty:', bads

    if isinstance(fig, list):
        for nr, figura in enumerate(fig):
            figura.savefig(os.path.join(OUTDIR, "ica_maps", filename+'_{}'.format(nr)+'.png'))
            pb.close(figura)
    else:
        fig.savefig(os.path.join(OUTDIR, "ica_maps", filename+'.png'))
        pb.close(fig)

    return ica, bads, events

def remove_ica_components(rm, ica, bads, events=[], scalings={'eeg': 4e-5, 'eog': 4e-5}, silent=False):
    """
    
    rm   - read manager with data to clean
    ica  - fitted mne.ica object to be used (e.g. returned by fit_eog_ica)
    bads - list of bad components of ICA
    events - detected eog events (nd.array)

    Returns ICA-corrected read manager
    """

    # read_manager to mne conversion
    raw = read_manager_continious_to_mne(rm)
    n = len(raw.ch_names)
    if not silent:
        raw.copy().plot(scalings=scalings, events=events, block=True, show=False, title='PRZED ICA',n_channels=n)

    raw_clean = ica.apply(raw, exclude=bads)

    if not silent:
        raw_clean.copy().plot(scalings=scalings, events=events, block=True, show=True, title='PO ICA',n_channels=n)

    data = np.array(raw_clean.to_data_frame())
    print "CONTROL INFO"
    print data.shape
    print np.median(np.abs(data), axis = 1)
    print np.std(data, axis = 1)

    # mne to read_manager conversion
    rm=deepcopy(rm)
    rm.set_samples(data.T, rm.get_param('channels_names'))

    return rm


def remove_eog_ica(rm, eog_chnl='eog',
                   montage=None, ds='',
                   use_eog_events=False, manual=False,
                   rejection_dict = dict(eeg=0.000150,
                                         eog=0.000250  # V
                                         ),
                   correlation_treshhold = 0.5,
                                         ):
                                         
    """
    Exists for compatibility reasons.
    rm - read manager with training data for ICA
    eog_chnl - channel to use as EOG source
    montage - montage of the read manager (for logging and filenames of generated images)
    use_eog_events True - split to EOG epochs - do ICA
    use eog events False - use whole file
    use_eog_events None - only create eog events
    manual True/False - shows ICA components, ICA components map
        prints correlations with EOG and then lets user write space seperated
        indexes of components to remove

    Returns
        - ICA-corrected read manager
        - detected eog events 
    """

    ica, bads, eog_events = fit_eog_ica(rm, eog_chnl, montage, ds, use_eog_events, manual, rejection_dict, correlation_treshhold)
    if bads:
        clean_rm = remove_ica_components(ica, bads, eog_events)
    else:
        clean_rm = rm
    
    return clean_rm, eog_events
    



def interp_bads(rm, bads):
    ds = read_manager_continious_to_mne(rm)
    ds.info['bads'] = bads
    ds.interpolate_bads()
    data = np.array(ds.to_data_frame())
    rm.set_samples(data.T*1e-6, rm.get_param('channels_names'))
    return rm

def select_bads_visually(rm, bad_chnls = [], no_plot=[], title=''):
    rm=deepcopy(rm)
    rm=mgr_filter(rm, [47.5, 52.5], [49.9, 50.1], 3, 25, ftype="cheby2", use_filtfilt=True)
    rmn = exclude_channels(rm, no_plot)
    ds = read_manager_continious_to_mne(rm)
    ds.drop_channels([str(i)for i in no_plot  if str(i) in ds.ch_names ])
    ds.info['bads'] = bad_chnls
#    ds.set_eeg_reference() #montaż CSA

    #ds.apply_proj()
    #dict(eeg=1e-8)
    n = len(ds.ch_names)
    ds.plot(scalings=dict(eeg=5e-4, eog=1e-3), block=True, show=True, title=title, highpass=0.5,lowpass=80,n_channels=n)
    print 'Selected bad chnls', ds.info['bads']
    return ds.info['bads']


def exclude_channels(mgr, channels):
    '''exclude all channels in channels list'''
    available = set(mgr.get_param('channels_names'))
    exclude = set(channels)
    channels = list(available.intersection(exclude))

    new_params = deepcopy(mgr.get_params())
    samples = mgr.get_samples()
    new_tags = deepcopy(mgr.get_tags())


    ex_channels_inds = [new_params['channels_names'].index(ch) for ch in channels]
    assert(-1 not in ex_channels_inds)

    new_samples = np.zeros((int(new_params['number_of_channels']) - len(channels),
                         len(samples[0])))
    # Define new samples and params list values
    keys = ['channels_names', 'channels_numbers', 'channels_gains', 'channels_offsets']
    keys_to_remove = []
    for k in keys:
        try:
            #Exclude from keys those keys that are missing in mgr
            mgr.get_params()[k]
        except KeyError:
            keys_to_remove.append(k)
            continue
        new_params[k] = []

    for k in keys_to_remove:
        keys.remove(k)
    new_ind = 0
    for ch_ind, ch in enumerate(samples):
        if ch_ind in ex_channels_inds:
            continue
        else:
            new_samples[new_ind, :] = ch
            for k in keys:
                new_params[k].append(mgr.get_params()[k][ch_ind])

            new_ind += 1

    # Define other new new_params
    new_params['number_of_channels'] = str(int(new_params['number_of_channels']) - len(channels))
    new_params['number_of_samples'] = str(int(new_params['number_of_samples']) - \
                                              len(channels)*len(samples[0]))


    info_source = read_info_source.MemoryInfoSource(new_params)
    tags_source = read_tags_source.MemoryTagsSource(new_tags)
    samples_source = read_data_source.MemoryDataSource(new_samples)
    return read_manager.ReadManager(info_source, samples_source, tags_source)


def leave_channels(mgr, channels):
    '''exclude all channels except those in channels list'''
    chans = deepcopy(mgr.get_param('channels_names'))
    for leave in channels:
        chans.remove(leave)
    return exclude_channels(mgr, chans)

def get_epochs_fromfile_flist(ds, tags_function_list, start_offset=-0.1, duration=2.0,
                        filters=[], decimate_factor=0, montage=None,
                        drop_chnls = [ u'AmpSaw', u'DriverSaw', u'trig1', u'trig2', u'Driver_Saw'],
                        tag_name = None,
                        gains = True,
                        tag_correction_chnls = [],
                        thr = None,
                        correction_window=[-0.1, 0.3],
                        correction_reverse = False,
                        tag_offset = 0.0,
                        remove_eog=None,
                        remove_artifacts=True,
                        bad_chnls_method = None,
                        bad_chnls_db='./bad_chnls_db.json',
                        bad_chnls=None,
                        get_last_tags=False,
                        ):
    '''For offline calibration and testing,
    Args:
        ds: dataset file name without extension.
        start_offset: baseline in negative seconds,
        duration: duration of the epoch (including baseline),
        filter: list of [wp, ws, gpass, gstop] for scipy.signal.iirdesign
                in Hz, Db, or None if no filtering is required
        montage: list of ['montage name', ...] ...-channel names if required
            montage name can be 'ears', 'csa', 'custom'
            ears require 2 channel names for ear channels
            custom requires list of reference channel names
        tags_function_list: list of tag filtering functions to get epochs for
        tag_correction_chnls: list of trigger channels, or empty list for no correction
        thr: trigger detection threshold in microvolts, None for standard deviation
        correction_window: list of 2 floats - window around tag to search in trigger channel
        tag_offset: float - seconds, to move all tags by this offset when using tag alignment
        correction_reverse: search for stimulation end in window, window MUST be long enough
        tag_name: tag name to be considered, if you want to use all
                  tags use None
        gains: True if you want to multiply signal by gains
        bad_chnls_method: method to deal with known bad channels (in some json dict or given as list)
                drop or interpolate ('drop', 'interp')
        bad_chnls_db: json dict with filenames as keys and list of channel name strings as values
        bad_chnls: if not None bad_chnls_db will be ignored and this list of strings will be used
        eog_clean_only: returns this function with cleaned readmanager
    Return: list of smarttags corresponding to tags_function_list'''
    eeg_rm = read_manager.ReadManager(ds+'.xml', ds+'.raw', ds+'.tag')
    if gains:
        eeg_rm.set_samples(get_microvolt_samples(eeg_rm),  eeg_rm.get_param('channels_names'))
    if tag_correction_chnls:
        align_tags(eeg_rm, tag_correction_chnls,
                    start_offset=correction_window[0],
                    duration=correction_window[1],
                    thr=thr,
                    reverse=correction_reverse,
                    offset=tag_offset)

    if bad_chnls_method is not None:
        if bad_chnls is None:
            with open(bad_chnls_db) as bcdbf:
                bad_channels_dict = json.load(bcdbf)
                try:
                    bad_chnls = bad_channels_dict[os.path.basename(ds) + '.raw']
                except KeyError:
                    bad_chnls = []
    if bad_chnls_method == 'drop':
        drop_chnls = drop_chnls + bad_chnls
        print '\n\n\nWILL DROP CHNLS', drop_chnls

    eeg_rm = exclude_channels(eeg_rm, drop_chnls)
    if bad_chnls_method == 'interp':
        eeg_rm = interp_bads(eeg_rm, bad_chnls)
    if decimate_factor:
        eeg_rm = mgr_decimate(eeg_rm, decimate_factor)
    for filter in filters:
        print "Filtering...", filter
        try:
            eeg_rm = mgr_filter(eeg_rm, filter[0], filter[1],filter[2],
                            filter[3], ftype=filter[4], use_filtfilt=True)
        except:
            print "Działanie filtrów uległo zmianie. Teraz należy podać listę list oraz podać typ filtru, np.:\n([[2, 29.6], [1, 45], 3, 20, 'cheby2'],[tu można podać kolejny filtr])"
            exit()
    if montage:
        if montage[0] == 'ears':
            eeg_rm = montage_ears(eeg_rm, montage[1], montage[2])
        elif montage[0] == 'csa':
            eeg_rm = montage_csa(eeg_rm)
        elif montage[0] == 'custom':
            eeg_rm = montage_custom(eeg_rm, montage[1:])
        else:
            raise Exception('Unknown montage')
    if remove_eog:
        ica, bads, eog_events = fit_eog_ica(eeg_rm, remove_eog, manual=True, montage=montage, ds=ds)
        if bads:
            eeg_rm = remove_ica_components(eeg_rm, ica, bads, eog_events, ds=ds)

    if remove_artifacts:
        pass
        #TODO dodać sprawdzenie, czy istnieje plik z artefaktami,
        #jeśli nie, to uruchomić tutaj funkcję, która go wygeneruje

        # eeg_rm = add_events_to_rm(eeg_rm, eog_events)
        # savetags_from_rm(eeg_rm, OUTDIR, postfix='EOG')

    #~ pb.plot(eeg_rm.get_samples()[5])
    #~ pb.show()
    if get_last_tags:
        tags = eeg_rm.get_tags()
        eeg_rm.set_tags(tags[-1-99:])
    tag_def = SmartTagDurationDefinition(start_tag_name=tag_name,
                                        start_offset=start_offset,
                                        end_offset=0.0,
                                        duration=duration)
    stags = SmartTagsManager(tag_def, '', '' ,'', p_read_manager=eeg_rm)
    returntags = []
    for tagfunction in tags_function_list:
        returntags.append(stags.get_smart_tags(p_func = tagfunction,))
    print 'Found epochs in defined groups:', [len(i) for i in returntags]
    return returntags



def GetEpochsFromRM(rm, tags_function_list, start_offset=-0.1, duration=2.0,
                        tag_name = None,
                        get_last_tags=False
                        ):
    '''Extracts stimulus epochs from ReadManager to list of SmartTags

    Args:
        rm: ReadManager with dataset
        start_offset: baseline in negative seconds,
        duration: duration of the epoch (including baseline),
        tags_function_list: list of tag filtering functions to get epochs for
        tag_name: tag name to be considered, if you want to use all tags use None
    Return:
        list of smarttags corresponding to tags_function_list'''


    if get_last_tags:
        tags = rm.get_tags()
        rm.set_tags(tags[-1-99:])
    tag_def = SmartTagDurationDefinition(start_tag_name=tag_name,
                                        start_offset=start_offset,
                                        end_offset=0.0,
                                        duration=duration)
    stags = SmartTagsManager(tag_def, '', '' ,'', p_read_manager=rm)
    returntags = []
    for tagfunction in tags_function_list:
        returntags.append(stags.get_smart_tags(p_func = tagfunction,))
    print 'Found epochs in defined groups:', [len(i) for i in returntags]
    return returntags



def evoked_from_smart_tags(tags, chnames, bas = -0.1):
    '''
    Args:
        tags: smart tag list, to average
        chnames: list of channels to use for averaging,
        bas: baseline (in negative seconds)'''
    min_length = min(i.get_samples().shape[1] for i in tags)
    # really don't like this, but epochs generated by smart tags can vary in length by 1 sample
    channels_data = []
    Fs = float(tags[0].get_param('sampling_frequency'))
    for i in tags:
        try:
            data = i.get_channels_samples(chnames)[:,:min_length]
        except IndexError: # in case of len(chnames)==1
            data = i.get_channels_samples(chnames)[None,:][:,:min_length]
        for nr, chnl in enumerate(data):
            data[nr] = chnl - np.mean(chnl[0:int(-Fs*bas)])# baseline correction
        if np.max(np.abs(data))<np.inf:
            channels_data.append(data)

    return np.mean(channels_data, axis=0), stats.sem(channels_data, axis=0)



def do_permutation_test(taglist, chnames, Fs, bas):
    """ between 2 conditions in taglist
    returns: list of clusters (per channel) with tuples (clusters, clusters_p_values)
    """

    print 'LEN TAGLIST =',len(taglist),"|",[len(t) for t in taglist] 
    min_length = min( [min(i.get_samples().shape[1] for i in tags) for tags in taglist])

    clusters = []  # per channel
    for channel in chnames:
        data_test = []
        print 'clustering for channel {}'.format(channel)
        for tags in taglist:
            data_tag = []
            for tag in tags:
                chnl_data = tag.get_channel_samples(channel)[:min_length]
                corrected_data = chnl_data - np.mean(chnl_data[0:int(-Fs * bas)])  # baseline correction
                data_tag.append(corrected_data)
            data_test.append(np.array(data_tag))
        if len(data_test)>1:
            T_obs, clusters_, cluster_p_values, H0 = mne.stats.permutation_cluster_test(data_test, step_down_p=0.05, n_jobs=8, seed=42)
        else: 
            T_obs, clusters_, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_test[0], step_down_p=0.05, n_jobs=8, seed=42)
        clusters.append((clusters_, cluster_p_values))
    return clusters


def evoked_list_plot_smart_tags(taglist, chnames=['O1', 'O2', 'Pz', 'PO7', 'PO8', 'PO3', 'PO4', 'Cz',],
                                start_offset=-0.1, labels=['target', 'nontarget'], show=True, size=(5,5),
                                addline=[], one_scale=True, anatomical=True, std_corridor=True, permutation_test=True):
    '''debug evoked potential plot,
     plot list of smarttags,
     blocks thread
     Args:
        taglist: list of smarttags
        labels: list of labels
        taglist, labels: lists of equal lengths,
        chnames: channels to plot
        start_offset: baseline in seconds
        addline: list of floats - seconds to add vertical barons.py", line 927, in do_autoreject
    segment_shape = ts.bad_segments.shape

        one_scale: binary - to force the same scale
        anatomical: plot all 10-20 electrodes with positions
        permutation_test: do a permutation test between target/nontarget
        '''

    fontsize=21
    figure_scale=8

    available_chnls = taglist[0][0].get_param('channels_names')
    chnames = [i for i in chnames if i in available_chnls]

    evs, stds = [], []
    for tags in taglist:

        ev, std = evoked_from_smart_tags(tags, chnames, start_offset)
        evs.append(ev)
        stds.append(std)

    Fs = float(taglist[0][0].get_param('sampling_frequency'))
    if permutation_test:
        clusters_per_chnl = do_permutation_test(taglist, chnames, bas=start_offset, Fs=Fs)


    times = []

    for ev in evs:
        time = np.linspace(0+start_offset, ev.shape[1]/Fs+start_offset, ev.shape[1])
        times.append(time)

    if one_scale:
        vmax = np.max(np.array(evs)+np.array(stds))
        vmin = np.min(np.array(evs)-np.array(stds))

    if anatomical:
        fig, axs = pb.subplots(5, 5, figsize=(1.*5*figure_scale, .5625*5*figure_scale))
        for ch in channels_not2draw(chnames):
            pos = map1020[ch]
            axs[pos.y, pos.x].axis("off")
        fig.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.97, wspace=0.16, hspace=0.28)
    else:
        fig = pb.figure(figsize=size)
    for nr, i in enumerate(chnames):
        if anatomical:
            pos = map1020[i]
            ax = axs[pos.y, pos.x]
        else:
            ax = fig.add_subplot( (len(chnames)+1)/2, 2, nr+1)
        if permutation_test:
            cl, p_val = clusters_per_chnl[nr]

            for cc, pp in zip(cl, p_val):
                ax.axvspan(times[0][cc[0].start], times[0][cc[0].stop - 1],
                                    color='blue' if pp<0.05 else 'gray', alpha=1-pp, zorder=1)

        for tagsnr in xrange(len(taglist)):
            color = None  # standard colors
            if 'target' in labels[tagsnr]:
                color = 'red'
            if 'nontarget' in labels[tagsnr]:  # sadly target is in nontarget, so no elsif
                color = 'green'

            if 'standard' in labels[tagsnr]:
                color = 'black'
            elif 'dewiant' in labels[tagsnr]:
                color = 'red'

            lines, = ax.plot(times[tagsnr],
                             evs[tagsnr][nr],
                             label = labels[tagsnr]+' N:{}'.format(len(taglist[tagsnr])),
                             color=color,
                             zorder=3
                             )
            ax.axvline(0, color='k')
            ax.axhline(0, color='k')
            for l in addline:
                ax.axvline(l, color='k')
            if std_corridor:
                ax.fill_between(times[tagsnr],
                                evs[tagsnr][nr]-stds[tagsnr][nr],
                                evs[tagsnr][nr]+stds[tagsnr][nr],
                                color = lines.get_color(),
                                alpha=0.3,
                                zorder=2
                                )
            if one_scale==True:
                ax.set_ylim(vmin, vmax)
            elif type(one_scale) == list:
                ax.set_ylim(one_scale[0], one_scale[1])
            ax.set_xlim(round(times[tagsnr][0],2), round(times[tagsnr][-1],2))
        ax.set_title(i)
        SetAxisFontsize(ax,fontsize*figure_scale/8)
    ax.legend(fontsize=fontsize*figure_scale/8)
    if show:
        pb.show()
    return fig



def mgr_decimate(mgr, factor):

    steps = int(factor/2)
    x = mgr.get_samples()
    for step in xrange(steps):
        if step == 0:
            new_samples = x
        y = signal.decimate(new_samples, 2, ftype='fir')
        new_samples = y
    info_source = deepcopy(mgr.info_source)
    info_source.get_params()['number_of_samples'] = new_samples.shape[1]
    info_source.get_params()['sampling_frequency'] = float(mgr.get_param('sampling_frequency'))/factor
    tags_source = deepcopy(mgr.tags_source)
    samples_source = read_data_source.MemoryDataSource(new_samples)
    return read_manager.ReadManager(info_source, samples_source, tags_source)

def mgr_order_filter(mgr, order=0, Wn=[49, 51], rp=None, rs=None, ftype='cheby2', btype='bandstop', output='ba', use_filtfilt=True, meancorr=1.0):
    nyquist = float(mgr.get_param('sampling_frequency'))/2.0
    if ftype in ['ellip','cheby2']:
        b,a = signal.iirfilter(order, np.array(Wn)/nyquist, rp, rs, btype=btype, ftype=ftype, output=output)
    else:
        b,a = signal.iirfilter(order, np.array(Wn)/nyquist, btype=btype, ftype=ftype, output=output)
    if use_filtfilt:
        for i in range(int(mgr.get_param('number_of_channels'))):
            mgr.get_samples()[i,:] = signal.filtfilt(b, a, mgr.get_samples()[i]-np.mean(mgr.get_samples()[i])*meancorr)
        samples_source = read_data_source.MemoryDataSource(mgr.get_samples(), False)
    else:
        print("FILTER CHANNELs")
        filtered = signal.lfilter(b, a, mgr.get_samples())
        print("FILTER CHANNELs finished")
        samples_source = read_data_source.MemoryDataSource(filtered, True)
    info_source = deepcopy(mgr.info_source)
    tags_source = deepcopy(mgr.tags_source)
    new_mgr = read_manager.ReadManager(info_source, samples_source, tags_source)
    return new_mgr

def mgr_filter(mgr, wp, ws, gpass, gstop, analog=0, ftype='ellip', output='ba', unit='hz', use_filtfilt=True, meancorr=1.0):
#    print "STOP"
#    exit()
    if unit == 'radians':
        b,a = signal.iirdesign(wp, ws, gpass, gstop, analog, ftype, output)
        w,h = signal.freqz(b,a,1000)
        
        fff = pb.figure()
        ax = fff.add_subplot()
        ax.plot(w,20*np.log10(abs(h)))
        pb.show()
    elif unit == 'hz':
        nyquist = float(mgr.get_param('sampling_frequency'))/2.0
        try:
            wp = wp/nyquist
            ws = ws/nyquist
        except TypeError:
            wp = [i/nyquist for i in wp]
            ws = [i/nyquist for i in ws]
        b,a = signal.iirdesign(wp, ws, gpass, gstop, analog, ftype, output) 
    if use_filtfilt:
        #samples_source = read_data_source.MemoryDataSource(mgr.get_samples(), False)
        for i in range(int(mgr.get_param('number_of_channels'))):
            #~ print("FILT FILT CHANNEL "+str(i))
            mgr.get_samples()[i,:] = signal.filtfilt(b, a, mgr.get_samples()[i]-np.mean(mgr.get_samples()[i])*meancorr)
        samples_source = read_data_source.MemoryDataSource(mgr.get_samples(), False)
    else:
        print("FILTER CHANNELs")
        filtered = signal.lfilter(b, a, mgr.get_samples())
        print("FILTER CHANNELs finished")
        samples_source = read_data_source.MemoryDataSource(filtered, True)
    info_source = deepcopy(mgr.info_source)
    tags_source = deepcopy(mgr.tags_source)
    new_mgr = read_manager.ReadManager(info_source, samples_source, tags_source)
    return new_mgr
#####


def _exclude_from_montage_indexes(mgr, chnames):
    exclude_from_montage_indexes = []

    for i in chnames:
        try:
            exclude_from_montage_indexes.append(mgr.get_param('channels_names').index(i))
        except ValueError:
            pass
    return exclude_from_montage_indexes


def montage_csa(mgr, exclude_from_montage=[]):
    exclude_from_montage_indexes = _exclude_from_montage_indexes(mgr, exclude_from_montage)
    new_samples = get_montage(mgr.get_samples(),
                              get_montage_matrix_csa(int(mgr.get_param('number_of_channels')),
                                                     exclude_from_montage=exclude_from_montage_indexes))
    info_source = deepcopy(mgr.info_source)
    tags_source = deepcopy(mgr.tags_source)
    samples_source = read_data_source.MemoryDataSource(new_samples)
    return read_manager.ReadManager(info_source, samples_source, tags_source)

def montage_ears(mgr, l_ear_channel, r_ear_channel, exclude_from_montage=[]):
    try:
        left_index = mgr.get_param('channels_names').index(l_ear_channel)
    except ValueError:
        print "Brakuje kanału usznego {}. Wykonuję montaż tylko do jegnego ucha.".format(l_ear_channel)
        return montage_custom(mgr, [r_ear_channel], exclude_from_montage)
    try:
        right_index = mgr.get_param('channels_names').index(r_ear_channel)
    except ValueError:
        print "Brakuje kanału usznego {}. Wykonuję montaż tylko do jegnego ucha.".format(r_ear_channel)
        return montage_custom(mgr, [l_ear_channel], exclude_from_montage)


    exclude_from_montage_indexes = _exclude_from_montage_indexes(mgr, exclude_from_montage)

    if left_index < 0 or right_index < 0:
        raise Exception("Montage - couldn`t find ears channels: "+str(l_ear_channel)+", "+str(r_ear_channel))

    new_samples = get_montage(mgr.get_samples(),
                              get_montage_matrix_ears(int(mgr.get_param('number_of_channels')),
                                                      left_index,
                                                      right_index,
                                                      exclude_from_montage_indexes
                                                      )
                              )
    info_source = deepcopy(mgr.info_source)
    tags_source = deepcopy(mgr.tags_source)
    samples_source = read_data_source.MemoryDataSource(new_samples)
    return read_manager.ReadManager(info_source, samples_source, tags_source)


def get_channel_indexes(channels, toindex):
    '''get list of indexes of channels in toindex list as found in
    channels list'''
    indexes = []
    for chnl in toindex:
        index = channels.index(chnl)
        if index<0:
            raise Exception("Montage - couldn`t channel: "+str(chnl))
        else:
            indexes.append(index)
    return indexes

def montage_custom(mgr, chnls, exclude_from_montage = []):
    '''apply custom montage to manager, by chnls'''

    exclude_from_montage_indexes = _exclude_from_montage_indexes(mgr, exclude_from_montage)

    indexes = []
    for chnl in chnls:
        print mgr.get_param('channels_names')
        index = mgr.get_param('channels_names').index(chnl)
        if index<0:
            raise Exception("Montage - couldn`t channel: "+str(chnl))
        else:
            indexes.append(index)

    new_samples = get_montage(mgr.get_samples(),
                              get_montage_matrix_custom(int(mgr.get_param('number_of_channels')),
                                                        indexes,
                                                        exclude_from_montage = exclude_from_montage_indexes,
                                                        )
                              )
    info_source = deepcopy(mgr.info_source)
    tags_source = deepcopy(mgr.tags_source)
    samples_source = read_data_source.MemoryDataSource(new_samples)
    return read_manager.ReadManager(info_source, samples_source, tags_source)

def get_montage(data, montage_matrix):
    """
    montage_matrix[i] = linear transformation of all channels to achieve _new_ channel i
    data[i] = original data from channel i

    >>> montage_matrix = np.array([[ 1.  , -0.25, -0.25, -0.25, -0.25], [-0.25,  1.  , -0.25, -0.25, -0.25], [-0.25, -0.25,  1.  , -0.25, -0.25],[-0.25, -0.25, -0.25,  1.  , -0.25], [-0.25, -0.25, -0.25, -0.25,  1.  ]])
    >>> data = np.array(5 * [np.ones(10)])
    >>> montage(data,montage_matrix)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])


    """

    return np.dot(montage_matrix, data)




def get_montage_matrix_csa(n, exclude_from_montage=[]):
    """
    Return nxn array representing extraction from
    every channel an avarage of all other channels.

    exclude_from_montage - list of indexes not to include in montage

    >>> get_montage_matrix_avg(5)
    array([[ 1.  , -0.25, -0.25, -0.25, -0.25],
           [-0.25,  1.  , -0.25, -0.25, -0.25],
           [-0.25, -0.25,  1.  , -0.25, -0.25],
           [-0.25, -0.25, -0.25,  1.  , -0.25],
           [-0.25, -0.25, -0.25, -0.25,  1.  ]])

    """


    n_fac = n-len(exclude_from_montage)
    factor = -1.0/(n_fac - 1)
    mx = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and not i in exclude_from_montage:
                mx[i, j] = factor
    return mx



def get_montage_matrix_ears(n, l_ear_index, r_ear_index, exclude_from_montage=[]):
    """
    Return nxn array representing extraction from
    every channel an avarage of channels l_ear_index
    and r_ear_index.

    exclude_from_montage - list of indexes not to include in montage

    >>> get_montage_matrix_ears(5, 2, 4)
    array([[ 1. ,  0. , -0.5,  0. , -0.5],
           [ 0. ,  1. , -0.5,  0. , -0.5],
               [ 0. ,  0. ,  1. ,  0. ,  0. ],
               [ 0. ,  0. , -0.5,  1. , -0.5],
               [ 0. ,  0. ,  0. ,  0. ,  1. ]])
    """

    factor = -0.5
    mx = np.diag([1.0]*n)
    for i in range(n):
        for j in range(n):
            if j in [r_ear_index, l_ear_index] \
                    and j != i \
                    and not i in [r_ear_index, l_ear_index] + exclude_from_montage:
                mx[i, j] = factor
    return mx

def get_montage_matrix_custom(n, indexes, exclude_from_montage=[]):
    """
    Return nxn array representing extraction from
    every channel an avarage of channels in indexes list

    exclude_from_montage - list of indexes not to include in montage

    >>> get_montage_matrix_custom(5, [2, 4])
    array([[ 1. ,  0. , -0.5,  0. , -0.5],
           [ 0. ,  1. , -0.5,  0. , -0.5],
               [ 0. ,  0. ,  1. ,  0. ,  0. ],
               [ 0. ,  0. , -0.5,  1. , -0.5],
               [ 0. ,  0. ,  0. ,  0. ,  1. ]])
    """

    factor = -1.0/len(indexes)
    mx = np.diag([1.0]*n)
    for i in range(n):
        for j in range(n):
            if j in indexes \
                    and j != i \
                    and not i in indexes + exclude_from_montage:
                mx[i, j] = factor
    return mx



def _do_autoreject(e_mne, fifname, interactive=True):
    ''' autoreject disabled, only interactive manual rejection'''
    ts = autoreject.LocalAutoRejectCV(consensus_percs=np.linspace(0, 0.99, 11))
        
#    clean_mne = ts.fit_transform(e_mne.pick_types(eeg=True))

    clean_mne = e_mne
    if interactive:
        #autoreject.plot_epochs(e_mne, bad_epochs_idx=ts.bad_epochs_idx,
        #        fix_log=ts.fix_log.as_matrix(), scalings='auto',
        #        title='', show=True, block=True)
        #print (clean_mne)
        clean_mne.plot(show=True, block=True, title='Select Remaining bad epochs to be dropped')
        clean_mne.plot(show=True, block=True, title='Please Double check')

        print (clean_mne)
    try:
        os.makedirs(os.path.join(OUTDIR, 'cache'))
    except Exception:
        pass
    clean_mne.save(fifname)
    return clean_mne, ts
    

def do_autoreject(epochs_list_local, get_from_file_args, clean_epochs_list, interactive, suffix, fname, dirty_epochs_list, use_cache=True):
    """
    does manual artefact rejection with caching
    
    :param epochs_list_local: a list of smart tags in epochs
    :param get_from_file_args: parameters of get_from_file function - for cache name
    :param clean_epochs_list: list of lists of clean epochs to be filled (by condition)
    :param suffix: string - suffix for cache file
    :param fname: name of first file in filelist
    :param dirty_epochs_list: dirty epoch list to fill with
    :param use_cache: To save selected clean signal to cache or to read that cache
    """
    fifname = os.path.join(OUTDIR, 'cache', os.path.basename(fname) + suffix + '_clean_epochs-epo.fif')

    cache_used = False
    if use_cache:
        try:
            clean_mne = mne.read_epochs(fifname)
            cache_used=True
        except Exception:
            cache_used=False

    if not cache_used:
        e_mne, tag_type_slices = read_manager_to_mne(epochs=epochs_list_local,
                                                    baseline=get_from_file_args['start_offset'])
        clean_mne, ts = _do_autoreject(e_mne, fifname, interactive)

        
        #bad_epochs_ids = ts.bad_epochs_idx
        dirty = []
        for type in epochs_list_local:
            dirty.extend(type)
        #dirty = [dirty[i] for i in bad_epochs_ids]
        #dirty_epochs_list.extend(dirty)
        #segment_shape = ts.bad_segments.shape
        #fig = pb.figure(figsize=(segment_shape[0]*0.2, segment_shape[1]*0.3))
        #ax = fig.add_subplot(111)
        
        #try:
        #    print('BAD SEGMENTS\n', ts.bad_segments)
        #    ax = sns.heatmap(ts.bad_segments.T, xticklabels=2, yticklabels=True, square=False,
        #                     cbar=False, cmap='Reds', ax = ax)
        #    ax.set_ylabel('Sensors')
        #    ax.set_xlabel('Trials')
        #    savemap = os.path.join(OUTDIR, os.path.basename(fname) + suffix + '_bad_epochs_map.png')
        #    fig.savefig(savemap)
        #    pb.close(fig)
        #except TypeError as e:
        #    print(traceback.format_exc())
        #    print(e)
        #    print('No map to save')


    clean_epochs_list_local = mne_to_rm_list(clean_mne)
    for epn, clean_epochs_local in enumerate(clean_epochs_list_local):
        clean_epochs_list[epn]+=clean_epochs_local
    return clean_epochs_list



def get_clean_epochs( filelist,
                      blok_type,
                      rej_amp = +np.inf,
                      rej_grad = +np.inf,
                      custom_suffix=None,
                      use_autoreject=True,
                      interactive = True,
                      autoreject_all=True,
                      autoreject_cache = False,
                      **get_epochs_fromfile_flist_args
                    ):

    tags_function_list = get_epochs_fromfile_flist_args['tags_function_list']
    montage = get_epochs_fromfile_flist_args['montage']
    bad_chnls_method = get_epochs_fromfile_flist_args['bad_chnls_method']
    bad_chnls_db = get_epochs_fromfile_flist_args['bad_chnls_db']
    
    
    print 'blok type', blok_type, 'montage', montage
    clean_epochs_list = [list() for tagtype in tags_function_list]
    #clean epochs list - list of smart tag lists, first dimension
    #is for every smarttag filtering function, second for every epoch
    if custom_suffix:
        suffix = custom_suffix
    else:
        if len(montage)<5:
            suffix = '_{}_blok_type-{}'.format(montage, blok_type)
        else:
            suffix = '_{}_blok_type-{}'.format(montage[:4]+['...'], blok_type)

    if bad_chnls_method == 'drop':  # we need to drop same channels for all files:
        bad_chnls = []
        with open(bad_chnls_db) as bad_chnls_db_f:
            bad_chnls_dict = json.load(bad_chnls_db_f)
            for fname in filelist:
                try:
                    bad_chnls.extend(bad_chnls_dict[os.path.basename(fname)])
                except KeyError:
                    pass
        get_epochs_fromfile_flist_args['bad_chnls'] = bad_chnls
        print '\n\n\nWILL DROP CHNLS', bad_chnls
    else:
        get_epochs_fromfile_flist_args['bad_chnls'] = None

    epochs_list_global = [list() for tagtype in tags_function_list] #lista epok dla wszystkich analizowanych plików

    for fname in filelist:
        if '.etr.' in fname:
            continue
        print "Lista plików! Plik:", fname
        ds=fname[:-4]
        get_epochs_fromfile_flist_args['ds'] = ds
        epochs_list_local = get_epochs_fromfile_flist(**get_epochs_fromfile_flist_args) #lista epok dla bieżącego pliku
        # z każdego pliku są listy dla dażdej filtrującej funkcji
        # brudne zapisujemy zaraz do pliku, czyste do jednej globalnej listy
        dirty_epochs_list = []
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            try:
                if use_autoreject: ### ??? ###
                    if autoreject_all: ### ??? ###
                        for nreps, eps in enumerate(epochs_list_local):
                            epochs_list_global[nreps].extend(eps)
                    else:
                        clean_epochs_list = artifacts_rejection(epochs_list_local, get_epochs_fromfile_flist_args, clean_epochs_list, interactive, suffix, fname, dirty_epochs_list, autoreject_cache)
                else:
                    for nr, epochs in enumerate(epochs_list_local):
                        #~ print epochs
                        clean, dirty = stag_amp_grad_art_rej(epochs, rej_amp, rej_grad)
                        if not clean:
                            print 'Warning, no clean epochs for defined groups'
                        clean_epochs_list[nr].extend(clean) # clean are different classes
                        dirty_epochs_list.extend(dirty) # dirty are for whole file
                    try:
                        os.mkdir(OUTDIR)
                    except Exception:
                        pass

                #zapisanie "brudnych" odcinków do pliku
                savefiletag = os.path.join(OUTDIR, os.path.basename(fname) + suffix + '_bad_epochs.tag')
                savetags(dirty_epochs_list, savefiletag,
                         get_epochs_fromfile_flist_args['start_offset'],
                         get_epochs_fromfile_flist_args['duration'])

            except Exception as e:
                print(traceback.format_exc())
                print e
                continue
    if use_autoreject and autoreject_all: # hash calculating and storing
        h = str(hex(hash(
                          str(filelist)
                        + str(get_epochs_fromfile_flist_args['montage'])
                        + str(get_epochs_fromfile_flist_args['filters'])
                        )
                    )
                )[-8:]
        fname_new = '_'.join(fname.split('_')[:4])+'_{}_'.format(h)
        clean_epochs_list = artifacts_rejection(epochs_list_global, get_epochs_fromfile_flist_args, clean_epochs_list, interactive, suffix, fname_new+"_ALL_", dirty_epochs_list, autoreject_cache)
    print 'Found clean epochs in defined groups:', [len(i) for i in clean_epochs_list]

    return clean_epochs_list


def Power(s,w,rng,Fs,ch=''):
    ps = []
    freq = np.fft.rfftfreq(w,1./Fs)
    a=np.argmin(np.abs(rng[0]-freq))
    b=np.argmin(np.abs(rng[1]-freq))
    if a==b: print "UWAGA! W podanym zakresie {} znalazła się tylko jedna częstość: {}".format(rng,freq[a])
    fs=[]
    window=signal.hanning(w)
    for i in range(0,s.size//w*w,w):
        f=np.fft.rfft(window*s[i:i+w],norm="ortho")
        p=np.mean(np.abs(f[a:b+1])**2)
        ps.append(p)

    ps=np.array(ps)
    return ps,(freq[a],freq[b])


def BandPower(rm,channels,width=2.,band=[49.5,50.5], thrs_type="-", thrs=[1e5,1e6,1e7]):
    thrs=np.array(thrs)
    Fs = int(float(rm.get_param('sampling_frequency')))
    width=int(width*Fs)+1
    T=np.arange(0,(float(rm.get_param('number_of_samples')))/Fs,1.*width/Fs)


    fig, axs = pb.subplots(5, 5, figsize=(5*5*2, 2.5*5*2))
    for ch in channels_not2draw(channels):
        pos = map1020[ch]
        axs[pos.y, pos.x].axis("off")
    
    A=[]
    Pgs=dict()
    for ch in channels:
        c = get_microvolt_samples(rm,ch)
        Pg,RealBand=Power(c,width,band,Fs,ch)
        
        Pgs[ch]=Pg
        A.append(Pg)
    A=np.array(A)

    fig.canvas.set_window_title("Energy in band [{:.2f}, {:.2f}]".format(*RealBand))

    ymax=np.max((np.max(A),thrs[2]))*2.
    ymin=np.min((np.min(A),thrs[0]))/2.
    if ymax>1e10: ymax=1e10

    for ch in channels:
        Pg=Pgs[ch]

        t=T[:Pg.size]
    
        pos = map1020[ch]
        ax = axs[pos.y, pos.x]
        ax.plot(t,Pg)
        ax.plot([t[0],t[-1]],[thrs[0],thrs[0]],'g--')
        ax.plot([t[0],t[-1]],[thrs[1],thrs[1]],'y--')
        ax.plot([t[0],t[-1]],[thrs[2],thrs[2]],'r--')
        
        ax.set_title(ch)
        ax.set_yscale('log')
        ax.set_ylim([ymin,ymax])
        ax.set_xlim((t[0],t[-1]))
        
        if thrs_type=="-": # below is good
            if np.sum(Pg>thrs[1])>0.1*Pg.size: ax.set_axis_bgcolor((0.95,0.95,0.65))
            if np.sum(Pg>thrs[2])>0.1*Pg.size: ax.set_axis_bgcolor((1.00,0.70,0.70))
            if Pg.max()<thrs[0]: ax.set_axis_bgcolor((0.80,1.00,0.80))
        if thrs_type=="+": # above is good
            if np.sum(Pg<thrs[1])>0.1*Pg.size: ax.set_axis_bgcolor((0.95,0.95,0.65))
            if np.sum(Pg<thrs[2])>0.1*Pg.size: ax.set_axis_bgcolor((1.00,0.70,0.70))
            if Pg.min()>thrs[0]: ax.set_axis_bgcolor((0.80,1.00,0.80))
        
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.97, wspace=0.16, hspace=0.28)
    return Pgs


def channels_not2draw(channels):
    '''zwraca listę kanałów, których brakuje w liście channels, a są w słowniku map1020'''
    ch_not2draw = []
    for ch in map1020:
        if ch not in channels: ch_not2draw.append(ch)
    return ch_not2draw
    
    
    
#tag filters for artifacts:
def outlier_tag_filter(tag):
    try: return tag['name']=='outlier'
    except: return False
def slope_tag_filter(tag):
    try: return tag['name']=='slope'
    except: return False
def slow_tag_filter(tag):
    try: return tag['name']=='slow'
    except: return False
def muscle_tag_filter(tag):
    try: return tag['name']=='muscle'
    except: return False


def PrepareRM(rm,drop_chnls=[],decimate_factor=0,filters=[],montage=[],gains=False):
    eeg_rm=deepcopy(rm)
    if gains:
        eeg_rm.set_samples(get_microvolt_samples(rm),  rm.get_param('channels_names'))
    eeg_rm = exclude_channels(eeg_rm, drop_chnls)
    if decimate_factor:
        eeg_rm = mgr_decimate(eeg_rm, decimate_factor)

    for filter in filters:
        print "Filtering...", filter
        try:
            eeg_rm = mgr_filter(eeg_rm, filter[0], filter[1],filter[2],
                            filter[3], ftype=filter[4], use_filtfilt=True)
        except IndexError:
            print "Działanie filtrów uległo zmianie. Teraz należy podać listę list oraz podać typ filtru, np.:\n([[2, 29.6], [1, 45], 3, 20, 'cheby2'],[tu można podać kolejny filtr])"
            exit()

    if montage:
        if montage[0] == 'ears':
            eeg_rm = montage_ears(eeg_rm, montage[1], montage[2], exclude_from_montage=["eog"])
        elif montage[0] == 'csa':
            eeg_rm = montage_csa(eeg_rm, exclude_from_montage=["eog"])
        elif montage[0] == 'custom':
            eeg_rm = montage_custom(eeg_rm, montage[1:], exclude_from_montage=["eog"])
        else:
            raise Exception('Unknown montage')
        
        if montage[1:]: #usunięcie ze zbioru kanałów użytych do montażu
            eeg_rm = exclude_channels(eeg_rm, montage[1:])
        
    return eeg_rm

def SetAxisFontsize(ax,size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)
