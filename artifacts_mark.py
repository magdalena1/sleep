#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Parametric Detection of EEG Artifacts
# Marcin Pietrzak 2017-03-01
# based on Klekowicz et al. DOI 10.1007/s12021-009-9045-2

import json
import os
import sys
from copy import deepcopy

import numpy as np

from helper_functions import PrepareRM, fit_eog_ica, remove_ica_components
from helper_functions import mgr_filter, get_microvolt_samples, Power,get_filelist
from obci.analysis.obci_signal_processing import read_manager
from obci.analysis.obci_signal_processing.tags.tags_file_writer import TagsFileWriter

OUTDIR = './'

def RM_ArtifactDetection(eeg_rm, drop_chnls, AllChannels, ArtDet_kargs={}):
    channels = [ 'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3',
                 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    available_chnls = eeg_rm.get_param('channels_names')

    #kanały, które będą testowane:
    Channels2BeTested = [c for c in channels if c in available_chnls]

    #filtrowanie dla zwykłych artefaktów:
    filters = [[0.5, 0.25, 3, 6.97, "butter"],
           [30, 60, 3, 12.4, "butter"],
           [[47.5,   52.5], [ 49.9,  50.1], 3, 25, "cheby2"]] 

    #filtrowanie dla detekcji mięśni:
    filters_muscle = [[0.5, 0.25, 3, 6.97, "butter"],
              [128, 192, 3, 12, "butter"]]
    line_fs = [50., 100., 150, 200]
    for line_f in line_fs: 
        filters_muscle.append([[line_f-2.5, line_f+2.5], [line_f-0.1, line_f+0.1], 3.0, 25., "cheby2"])

    #jeśli którys z poniższych parametrów nie został podany ręcznie, to jest używana wartość domyślna
    forget = ArtDet_kargs.get('forget',2.) # [s] długość fragmentów na początku i końcu sygnału, dla których nie zapisujemy tagów

    SlopeWindow = ArtDet_kargs.get('SlopeWindow',0.07)       # [s] #szerokość okna, w którym szukamy iglic i stromych zboczy (slope)
    SlowWindow  = ArtDet_kargs.get('SlowWindow', 1)          # [s] #szerokość okna, w którym szukamy fal wolnych (slow)
    OutliersMergeWin =ArtDet_kargs.get('OutliersMergeWin',.1)# [s] #szerokość okna, w którym łączymy sąsiadujące obszary z próbkami outlier
    MusclesFreqRange = ArtDet_kargs.get('MusclesFreqRange',[40,250]) # [Hz] przedział częstości, w którym szukamy artefaktów mięśniowych

    #progi powyżej których próbka jest oznaczana, jako slope:
    SlopesAbsThr = ArtDet_kargs.get('SlopesAbsThr', 50)       # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlopeWindow
    SlopesStdThr = ArtDet_kargs.get('SlopesStdThr', 6)        # wartość peak2peak w oknie, jako wielokrotność std
    SlopesThrRel2Med = ArtDet_kargs.get('SlopesThrRel2Med', 6)# wartość peak2peak w oknie, jako wielokrotność mediany
    SlopesThrs = (SlopesAbsThr, SlopesStdThr, SlopesThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako slow (fala wolna):
    SlowsAbsThr = ArtDet_kargs.get('SlowsAbsThr', 80)       # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlowWindow
    SlowsStdThr = ArtDet_kargs.get('SlowsStdThr', 4)        # wartość peak2peak w oknie, jako wielokrotność std
    SlowsThrRel2Med = ArtDet_kargs.get('SlowsThrRel2Med', 3)# wartość peak2peak w oknie, jako wielokrotność mediany
    SlowsThrs = (SlowsAbsThr, SlowsStdThr, SlowsThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako outlier:
    OutliersAbsThr = ArtDet_kargs.get('OutliersAbsThr', 100)       # [µV] bezwzględna wartość amplitudy (liczona od 0)
    OutliersStdThr = ArtDet_kargs.get('OutliersStdThr', 8)         # wartość amplitudy, jako wielokrotność std
    OutliersThrRel2Med = ArtDet_kargs.get('OutliersThrRel2Med', 16)# wartość amplitudy, jako wielokrotność mediany
    OutliersThrs = (OutliersAbsThr, OutliersStdThr, OutliersThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako muscle:
    MusclesAbsThr = ArtDet_kargs.get('MusclesAbsThr', np.inf)   # [µV²] bezwzględna wartość średniej mocy na próbkę
                                                                # w zakresie częstości MusclesFreqRange, której lepiej nie ustawiać,
                                                                # bo pod tym wzlęgem kanały się bardzo różnią
    MusclesStdThr = ArtDet_kargs.get('MusclesStdThr', 5)        # wartość amplitudy, jako wielokrotność std
    MusclesThrRel2Med = ArtDet_kargs.get('MusclesThrRel2Med', 5)# wartość amplitudy, jako wielokrotność mediany
    MusclesThrs = (MusclesAbsThr, MusclesStdThr, MusclesThrRel2Med)

    #UWAGA! progi odnoszące się do std lub mediany są szacowane dla każdego kanału osobno
    # tagi są zaznaczane wg najostrzejszego z kryteriów
    # żeby nie korzystać z danego kryterium wystarczy podać wartość np.inf
    # każdy tag ma zapisane informacje:
    #   które kryterium było wykorzystane (typ:'abs', 'std' lub 'med')
    #   jaki był ostateczny próg (thr:wartość) w µV
    #   jaka wartość przekroczyła próg (val:wartość) w µV

    return detect_artifacts(eeg_rm, Channels2BeTested, AllChannels, filters, filters_muscle, forget, OutliersMergeWin, SlopeWindow, SlowWindow, MusclesFreqRange, SlopesThrs, SlowsThrs, OutliersThrs, MusclesThrs)


def detect_artifacts(eeg_rm, channels, AllChannels, filters, filters_muscle, forget, OutliersMergeWin, SlopeWindow, SlowWindow, MusclesFreqRange, SlopesThrs, SlowsThrs, OutliersThrs, MusclesThrs):

    SlopesAbsThr, SlopesStdThr, SlopesThrRel2Med = SlopesThrs
    SlowsAbsThr, SlowsStdThr, SlowsThrRel2Med = SlowsThrs
    OutliersAbsThr, OutliersStdThr, OutliersThrRel2Med = OutliersThrs
    MusclesAbsThr, MusclesStdThr, MusclesThrRel2Med = MusclesThrs

    Fs = float(eeg_rm.get_param('sampling_frequency'))

    tags=[]

    bad_samples,thresholds=outliers(eeg_rm, channels, filters, StdThr=OutliersStdThr,
                AbsThr=OutliersAbsThr,ThrRel2Med=OutliersThrRel2Med) #zaznaczanie outliers
    bad_samples=ForgetBirthAndDeath(bad_samples,Fs, forget) #zapominamy wadliwe próbki na początku i końcu sygnału
    tags.extend(ArtTagsWriter(bad_samples, Fs, AllChannels, MergWin=OutliersMergeWin, ArtName="outlier",Thresholds=thresholds))

    bad_samples,thresholds=P2P(eeg_rm, channels, SlowWindow, filters, StdThr=SlowsStdThr,
                AbsThr=SlowsAbsThr, ThrRel2Med=SlowsThrRel2Med) #zaznaczanie fal wolnych
    bad_samples=ForgetBirthAndDeath(bad_samples,Fs, forget) #zapominamy wadliwe próbki na początku i końcu sygnału
    tags.extend(ArtTagsWriter(bad_samples, Fs, AllChannels, MergWin=SlowWindow, ArtName="slow",Thresholds=thresholds))

    bad_samples,thresholds=P2P(eeg_rm, channels, SlopeWindow, filters, StdThr=SlopesStdThr, 
                AbsThr=SlopesAbsThr, ThrRel2Med=SlopesThrRel2Med) #zaznaczanie iglic i stromych zboczy
    bad_samples=ForgetBirthAndDeath(bad_samples,Fs, forget) #zapominamy wadliwe próbki na początku i końcu sygnału
    tags.extend(ArtTagsWriter(bad_samples, Fs, AllChannels, MergWin=SlopeWindow, ArtName="slope",Thresholds=thresholds))

    bad_samples,thresholds=muscles(eeg_rm, channels, SlowWindow, MusclesFreqRange, filters_muscle, StdThr=MusclesStdThr, 
                AbsThr=MusclesAbsThr, ThrRel2Med=MusclesThrRel2Med) #zaznaczanie mięśni
    bad_samples=ForgetBirthAndDeath(bad_samples,Fs, forget) #zapominamy wadliwe próbki na początku i końcu sygnału
    tags.extend(ArtTagsWriter(bad_samples, Fs, AllChannels, MergWin=SlopeWindow, ArtName="muscle",Thresholds=thresholds))

#    to będzie musiało być w innym miejscu (po akceptacji przez użytkownika)
#    ArtTagsSaver(ArtifactsFilePath,tags) #zapis tagów do pliku

    return tags


def GetDirtyEpochsIDs(epochs_list, start_offset, duration, artifact_tags, eeg_rm=None, drop_chnls=None, AllChannels=None,  ArtDet_kargs={}):
    #return ids of epochs with artifacts
    if not artifact_tags:
        artifact_tags = RM_ArtifactDetection(eeg_rm, drop_chnls, AllChannels, ArtDet_kargs)
    #tablica granic wszystkich artefaktów bez względu na kanał:
    ARTS={'outlier':0,'slow':1,'slope':2,'muscle':3}
    COLORS=[(0.9, 0., 0., 1.),(0.6, 0., 0.6, 1.),(0.6, 0.6, 0., 1.),(0.1, 0.1, 0.8, 1.)]
    ArtifactsBorders = np.array( [[-np.inf,-np.inf,0]]+[[tag['start_timestamp'], tag['end_timestamp'],ARTS[tag['name']]] for tag in artifact_tags] )
    #pętla po odcinkach z bodźcami i zapisywanie numerów tych, w których wystąpił jakiś artefakt:
    dirty_epochs_ids={}
    epoch_id=-1
    for type_id,epoch_by_type in enumerate(epochs_list):
        for epoch in epoch_by_type:
            epoch_id+=1
            #początek i koniec odcinka, który badamy
            start = float(epoch.get_start_timestamp())-start_offset
            end = start+duration

            artifacts_starting_before_end = ArtifactsBorders[:,0]<end
            artifacts_ending_after_start  = ArtifactsBorders[:,1]>start
            #wystarczy każde, choćby niewielkie przekrycie

            artifacts_overlapping = artifacts_starting_before_end*artifacts_ending_after_start

            if np.any(artifacts_overlapping):
                ID = np.argmax(artifacts_overlapping)
                dirty_epochs_ids[epoch_id]=COLORS[int(ArtifactsBorders[ID,2])]
    return dirty_epochs_ids, artifact_tags


def artifacts_epochs_cleaner(epochs_list,artifact_tags, start_offset, duration):
    #reject epochs with artifacts

    #tablica granic wszystkich artefaktów bez względu na kanał:
    ArtifactsBorders = np.array( [[tag['start_timestamp'], tag['end_timestamp']] for tag in artifact_tags] )

    #pętla po odcinkach z bodźcami i zapisywanie tylko tych, w których nie wystąpił żaden artefakt:
    cleaned_epochs=[[] for e in epochs_list]
    for type_id,epoch_by_type in enumerate(epochs_list):
        for epoch in epoch_by_type:

            #początek i koniec odcinka, który badamy
            start = float(epoch.get_start_timestamp())-start_offset
            end = start+duration

            artifacts_starting_before_end = ArtifactsBorders[:,0]<end
            artifacts_ending_after_start  = ArtifactsBorders[:,1]>start
            #wystarczy każde, choćby niewielkie przekrycie

            artifacts_overlapping = artifacts_starting_before_end*artifacts_ending_after_start

            if not np.any(artifacts_overlapping):
                cleaned_epochs[type_id].append(epoch)

    print "Number of epochs types:",len(cleaned_epochs)
    for i in range(len(epochs_list)):
        print "clean epochs: {} of {} stimulus".format(len(cleaned_epochs[i]),len(epochs_list[i]))
#    raw_input("Press Enter to continue")

    return cleaned_epochs



def ForgetBirthAndDeath(bs,Fs,forget=1.):
    for ch in bs:
        bs[ch][:int(Fs*forget)]=0.0
        bs[ch][-int(Fs*forget):]=0.0
    return bs


def outliers(rm, channels, filters=[], StdThr=np.inf, AbsThr=np.inf, ThrRel2Med=np.inf):
    #w gruncie rzeczy to jeszcze można zrobić test, czy rozkład jest gaussowski,
    #bo czasami nie jest i wtedy można sugerować kanał, jako wadliwy
    #może to m.in. sugerować, że właściwości statystyczne kanału nie są stałe w czasie, np. odkleił się
    
    rm=deepcopy(rm)
    for filter in filters:
        print "Filtering...", filter
        try:
            rm = mgr_filter(rm, filter[0], filter[1],filter[2],
                            filter[3], ftype=filter[4], use_filtfilt=True)
        except:
            print "Działanie filtrów uległo zmianie. Teraz należy podać listę list oraz podać typ filtru, np.:\n([[2, 29.6], [1, 45], 3, 20, 'cheby2'],[tu można podać kolejny filtr])"

    Fs = int(float(rm.get_param('sampling_frequency')))
    bad_samples=dict()
    Thresholds=dict()
    print channels
    for ch in channels:
        c = get_microvolt_samples(rm,ch)
        c-=c.mean()
        
        #obliczenie progu dla std
        C=c.copy()
        s=np.inf
        while C.std()<s:
            s=C.std()
            C=C[np.abs(C)<StdThr*s+C.mean()]
        
        Threshold = min(AbsThr,ThrRel2Med*np.median(np.abs(c)), StdThr*s) #wybieramy najostrzejszy (najniższy) próg
        ThresholdType = ['abs','med','std'][np.argmin((AbsThr,ThrRel2Med*np.median(np.abs(c)), StdThr*s))]
        print AbsThr,ThrRel2Med*np.median(np.abs(c)), StdThr*s,ThresholdType
        Thresholds[ch]=[Threshold,ThresholdType]
        
        if Threshold<=0:
            raise Exception("Threshold = {}\nYou must give some threshold in range (0,+oo)!".format(Threshold))

        bad_samples[ch]=(np.abs(c)>Threshold)*np.abs(c)

    return bad_samples, Thresholds


def P2P(rm, channels, width, filters=[], StdThr=np.inf, AbsThr=np.inf, ThrRel2Med=np.inf):
    rm=deepcopy(rm)
    for filter in filters:
        print "Filtering...", filter
        try:
            rm = mgr_filter(rm, filter[0], filter[1],filter[2],
                            filter[3], ftype=filter[4], use_filtfilt=True)
        except:
            print "Działanie filtrów uległo zmianie. Teraz należy podać listę list oraz podać typ filtru, np.:\n([[2, 29.6], [1, 45], 3, 20, 'cheby2'],[tu można podać kolejny filtr])"

    Fs = int(float(rm.get_param('sampling_frequency')))
    width=int(width*Fs)
    bad_samples=dict()
    Thresholds=dict()
    for ch in channels:
        c = get_microvolt_samples(rm,ch)
        
        mm=MiniMax(c,width,ch)
        #przesuniecie o polowe szerokosci okna i ponowne obliczenie
        mm_2=MiniMax(np.roll(c,-width/2),width,ch)

        #obliczenie progu dla std
        MM=np.concatenate([mm,mm_2])
        s=np.inf
        while MM.std()<s:
            s=MM.std()
            MM=MM[MM<StdThr*s+MM.mean()]

        Threshold_candidates=[AbsThr,ThrRel2Med*np.median([mm,mm_2]), StdThr*s+MM.mean()]
        Threshold = min(Threshold_candidates) #wybieramy najostrzejszy (najnizszy) prog
        ThresholdType = ['abs','med','std'][np.argmin(Threshold_candidates)]
        print ch,Threshold_candidates,ThresholdType
        Thresholds[ch]=[Threshold,ThresholdType]
        
        if Threshold<=0:
            raise Exception("Threshold = {}\nYou must give some threshold in range (0,+oo)!".format(Threshold))
        bad_segment=mm*(mm>Threshold) #jednostronnie, bo interesują nas tylko przypadki o zbyt dużej, a nie zbyt małej amplitudzie
        bad_segment_2=mm_2*(mm_2>Threshold) #jednostronnie przesunięte
        
        bad_samples[ch]=np.max([BadSeg2BadSamp(bad_segment,width,c.size),BadSeg2BadSamp(bad_segment_2,width,c.size,width/2)],axis=0)

#        pb.plot(c[bad_samples[ch]>0])
#        pb.plot(bad_samples[ch][bad_samples[ch]>0],'r')
#        pb.title(ch)
#        pb.show()
    return bad_samples, Thresholds

def BadSeg2BadSamp(bad_segments,width,size,shift=0):
    y=np.zeros(size,dtype=bad_segments.dtype)
    for i in range(bad_segments.size):
        y[i*width+shift:(i+1)*width+shift]=bad_segments[i]
    return y


def ArtTagsWriter(bad_samples,Fs,chNs,MergWin,ArtName='artifact',Thresholds=np.inf,MinDuration=0.03):
    tags=[]
    for ch in bad_samples:
        if Thresholds==np.inf: threshold,threshold_type=np.inf,'art' #default
        else: threshold,threshold_type=Thresholds[ch]

        bads=bad_samples[ch]
        start=stop=0.
        artifact=False
        value=-np.inf
        for i in range(bads.size):
            if bads[i]:
                if bads[i]>value: value=bads[i]
                stop=i
                if not artifact:
                    start=i
                    artifact=True
            elif artifact:
                if i-stop>MergWin*Fs:
                    if stop-start<MinDuration*Fs: #symetryczne rozszerzenie tagu, żeby nie był węższy niż MinDuration
                        Ext=MinDuration*Fs-stop+start
                        stop+=Ext/2
                        start-=Ext/2
                    tag={ 'channelNumber':chNs.index(ch),'start_timestamp': 1.*start/Fs, 'name': ArtName, 'end_timestamp': 1.*stop/Fs, 'desc': {'thr': "{}".format(int(threshold)), 'val': "{}".format(int(value)), 'typ':threshold_type}}
                    tags.append(tag)
                    value=-np.inf
                    artifact=False

    return tags

def ArtTagsSaver(ArtifactsFilePath,tags, rej_dict):
    try: os.mkdir(os.path.dirname(ArtifactsFilePath))
    except: pass
    writer = TagsFileWriter(ArtifactsFilePath)
    for tag in tags:
        writer.tag_received(tag)
    writer.finish_saving(0.0)

    rej_dict_file = open(ArtifactsFilePath[0:-9]+'_artifact_rej_dict.json', 'w')
    json.dump(rej_dict, rej_dict_file, indent=4, sort_keys=True)
    rej_dict_file.close()


def MiniMax(s,w,ch=''):
    mm = []
    for i in range(0,s.size,w):
        m=s[i:i+w].max()-s[i:i+w].min()
        mm.append(m)
    return np.array(mm)



def muscles(rm, channels, width, rng, filters=[], StdThr=np.inf, AbsThr=np.inf, ThrRel2Med=np.inf):
    rm=deepcopy(rm)
    for filter in filters:
        print "Filtering...", filter
        try:
            rm = mgr_filter(rm, filter[0], filter[1],filter[2],
                            filter[3], ftype=filter[4], use_filtfilt=True)
        except:
            print "Działanie filtrów uległo zmianie. Teraz należy podać listę list oraz podać typ filtru, np.:\n([[2, 29.6], [1, 45], 3, 20, 'cheby2'],[tu można podać kolejny filtr])"

    Fs = int(float(rm.get_param('sampling_frequency')))
    width=int(width*Fs)
    bad_samples=dict()
    Thresholds=dict()
    for ch in channels:
        c = get_microvolt_samples(rm,ch)
        c-=c.mean()
        
        Pmsc,freqs=Power(c,width,rng,Fs,ch)
        #przesunięcie o połowę szerokości okna i ponowne obliczenie
        Pmsc_2,freqs=Power(np.roll(c,-width/2),width,rng,Fs)
        
        #obliczenie progu dla std
        PP=np.concatenate([Pmsc,Pmsc_2])
        s=np.inf
        while PP.std()<s:
            s=PP.std()
            PP=PP[PP<StdThr*s+PP.mean()]

        Threshold_candidates=[AbsThr,ThrRel2Med*np.median([Pmsc,Pmsc_2]), StdThr*s+PP.mean()]
        Threshold = min(Threshold_candidates) #wybieramy najostrzejszy (najnizszy) prog
        ThresholdType = ['abs','med','std'][np.argmin(Threshold_candidates)]
        print ch,Threshold_candidates,ThresholdType
        Thresholds[ch]=[Threshold,ThresholdType]
        
        if Threshold<=0:
            raise Exception("Threshold = {}\nYou must give some threshold in range (0,+oo)!".format(Threshold))

        bad_segment=Pmsc*(Pmsc>Threshold) #jednostronnie, bo interesują nas tylko przypadki o zbyt dużej, a nie zbyt małej mocy
        bad_segment_2=Pmsc_2*(Pmsc_2>Threshold) #jednostronnie
        
        bad_samples[ch]=np.max([BadSeg2BadSamp(bad_segment,width,c.size),BadSeg2BadSamp(bad_segment_2,width,c.size,width/2)],axis=0)

    return bad_samples, Thresholds




def main(argv): #to raczej do testowania niż faktycznego używania i może nawet obecnie nie działać ze względu na zmiany, jakie zaszły
    filelist, filename = get_filelist(argv[0], argv[1:])

    bad_chnls_method = "drop"

    #kanały, które będą testowane:
    channels = [ 'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3',
                 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']

    # kanały, które nie zostaną użyte w żaden sposób, bo nie zawierają zapiu EEG
    drop_chnls = [u'l_reka', u'p_reka', u'l_noga', u'p_noga', u'haptic1', u'haptic2',
                  u'phones', u'Driver_Saw', u'Saw', u'Sample_Counter']

    #filtrowanie dla zwykłych artefaktów:
    filters = [[0.5, 0.25, 3, 6.97, "butter"],
           [30, 60, 3, 12.4, "butter"],
           [[47.5,   52.5], [ 49.9,  50.1], 3, 25, "cheby2"]] 

    filters = [[[2, 29.6], [1, 45], 3, 20, "cheby2"]] # tu można podać kilka różnych filtrów, zostaną zastosowane kolejno

#    filters = [[1, 0.25, 3, 12.28, "butter"],
#           [30, 60, 3, 12.4, "butter"],
#           [[47.5,   52.5], [ 49.9,  50.1], 3, 25, "cheby2"]] 

#    filters = [[2, 0.62, 3, 10.4, "butter"],
#           [30, 60, 3, 12.4, "butter"],
#           [[47.5,   52.5], [ 49.9,  50.1], 3, 25, "cheby2"]] 

    #filtrowanie dla detekcji mięśni:
    filters_muscle = [[0.5, 0.25, 3, 6.97, "butter"],
              [128, 192, 3, 12, "butter"]]
    line_fs = [50., 100., 150, 200]
    for line_f in line_fs: 
        filters_muscle.append([[line_f-2.5, line_f+2.5], [line_f-0.1, line_f+0.1], 3.0, 25., "cheby2"])

    forget = 2. # [s] długość fragmentów na początku i końcu sygnału, dla których nie zapisujemy tagów

    SlopeWindow = 0.07 # [s] #szerokość okna, w którym szukamy iglic i stromych zboczy (slope)
    SlowWindow  = 1    # [s] #szerokość okna, w którym szukamy fal wolnych (slow)
    MusclesFreqRange = [40,250] # [Hz] przedział częstości, w którym szukamy artefaktów mięśniowych

    #progi powyżej których próbka jest oznaczana, jako slope:
    SlopesAbsThr      = 50   # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlopeWindow
    SlopesStdThr      = 5    # wartość peak2peak w oknie, jako wielokrotność std
    SlopesThrRel2Med  = 5   # wartość peak2peak w oknie, jako wielokrotność mediany
    SlopesThrs = (SlopesAbsThr, SlopesStdThr, SlopesThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako slow (fala wolna):
    SlowsAbsThr      = 75   # [µV] bezwzględna wartość amplitudy peak2peak w oknie SlowWindow
    SlowsStdThr      = 3    # wartość peak2peak w oknie, jako wielokrotność std
    SlowsThrRel2Med  = 2   # wartość peak2peak w oknie, jako wielokrotność mediany
    SlowsThrs = (SlowsAbsThr, SlowsStdThr, SlowsThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako outlier:
    OutliersAbsThr      = 100   # [µV] bezwzględna wartość amplitudy (liczona od 0)
    OutliersStdThr      = 8    # wartość amplitudy, jako wielokrotność std
    OutliersThrRel2Med  = 16    # wartość amplitudy, jako wielokrotność mediany
    OutliersThrs = (OutliersAbsThr, OutliersStdThr, OutliersThrRel2Med)

    #progi powyżej których próbka jest oznaczana, jako muscle:
    MusclesAbsThr      = np.inf   # [µV²] bezwzględna wartość średniej mocy na próbkę w zakresie częstości MusclesFreqRange
                                 # lepiej nie ustawiać, bo pod tym wzlęgem kanały się bardzo różnią (bo różnie szumią)
    MusclesStdThr      = 5    # wartość amplitudy, jako wielokrotność std
    MusclesThrRel2Med  = 5    # wartość amplitudy, jako wielokrotność mediany
    MusclesThrs = (MusclesAbsThr, MusclesStdThr, MusclesThrRel2Med)

    #UWAGA! progi odnoszące się do std lub mediany są szacowane dla każdego kanału osobno
    # tagi są zaznaczane wg najostrzejszego z kryteriów
    # żeby nie korzystać z danego kryterium wystarczy podać wartość np.inf
    # każdy tag ma zapisane informacje:
    #   które kryterium było wykorzystane (typ:'abs', 'std' lub 'med')
    #   jaki był ostateczny próg (thr:wartość) w µV
    #   jaka wartość przekroczyła próg (val:wartość) w µV

    bad_chnls = []

    for fname in filelist:
        if '.etr.' in fname:
            continue
        print "Lista plików! Plik:", fname
        ds=fname[:-4]

        eeg_rm = read_manager.ReadManager(ds+'.xml', ds+'.raw', ds+'.tag')
        AllChannels=eeg_rm.get_param('channels_names') #all channels in dataset saved for future use

        ArtifactsFilePath=os.path.join(OUTDIR, 'cache',"{}_{}_{}".format(os.path.basename(fname).split(".")[0],"solo","artifacts.obci.tag"))
        
        detect_artifacts(eeg_rm, ArtifactsFilePath, channels, AllChannels, filters, filters_muscle, forget, SlopeWindow, SlowWindow, MusclesFreqRange, SlopesThrs, SlowsThrs, OutliersThrs, MusclesThrs)



if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except IndexError:
        raise IOError( 'Please provide path to .raw')




