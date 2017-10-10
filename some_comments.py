	# t = tags_reader.TagsFileReader(tags_file)
	# tags = t.get_tags()

	# nyq = fs/2.0
	# # [n, Wn] = ss.cheb2ord(0.5 / nyq, 0.2 / nyq, 3, 12, analog=0)
	# # [b_high, a_high] = ss.cheby2(n, 12, Wn, btype='high', analog=0, output='ba')

	# [n, Wn] = ss.buttord(0.3 / nyq, 0.1 / nyq, 3, 12, analog=0)
	# [b_high, a_high] = ss.butter(n, Wn, btype='high', analog=0, output='ba')
	# # #plot impulse reponse
	# # w, h = ss.freqz(b_high, a_high)
	# # w = w * fs / (2 * np.pi) #in Hz
	# # py.plot(w, 20 * np.log10(abs(h)), 'b')

	# [n, Wn] = ss.buttord(2.0 / nyq, 2.3 / nyq, 3, 12, analog=0)
	# [b_low, a_low] = ss.butter(n, Wn, btype='low', analog=0, output='ba')
	# # w, h = ss.freqz(b_low, a_low)
	# # w = w * fs / (2 * np.pi) #in Hz
	# # py.plot(w, 20 * np.log10(abs(h)), 'b')

	# signal_swa = ss.filtfilt(b_low, a_low, signal[0,:]) 
	# signal_swa = ss.filtfilt(b_high, a_high, signal_swa) 

	# [n, Wn] = ss.buttord(10.5 / nyq, 8.5 / nyq, 3, 12, analog=0)
	# [b_high, a_high] = ss.butter(n, Wn, btype='high', analog=0, output='ba')

	# [n, Wn] = ss.buttord(16.0 / nyq, 18.0 / nyq, 3, 12, analog=0)
	# [b_low, a_low] = ss.butter(n, Wn, btype='low', analog=0, output='ba')

	# signal_ss = ss.filtfilt(b_low, a_low, signal[0,:]) 
	# signal_ss = ss.filtfilt(b_high, a_high, signal_ss) 
	# ss_envelope = np.abs(ss.hilbert(signal_ss))

	# rms_ampli = scoreatpercentile(rms, 99.0)*np.sqrt(2) 
	# mad = np.median(np.abs(rms-np.median(rms))) #median absolute deviation
	# # new_rms = np.delete(rms,np.where(rms>20*mad)[0])
	# # rms_ampli = scoreatpercentile(new_rms, 95.0)*np.sqrt(2) 

	# ### perform 1/f^a curve fitting
	# window = int(10 * fs) #in samples
	# df = 1./2
	# f_limit = 40 #in Hz
	# spectra = np.zeros((int(signal.shape[1]/window-1), int(1/df*f_limit+1)))
	# j = 0
	# for i in xrange(0, signal.shape[1] - window, int(window)):
	#     s = signal[0,i:i+window]
	#     f, pxx = ss.welch(s, fs=fs, window='hanning', nperseg=1./df*fs, noverlap=None, nfft=None, detrend='constant', return_onesided=True)
	#     spectra[j, :] = np.log10(pxx[:np.where(f==f_limit)[0][0]+1]) 
	#     j += 1
	# popt, pcov = curve_fit(spectrum_model, f[:int(1/df*f_limit+1)], spectra[0,:])

	# ###

	#theta, alfa, beta

		# chosen_KC = filter_atoms_KC(f_name+'_KC.csv', epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [0.035,2.5], [0.3,1.5], 1.0, [140,600],[-0.5,0.5])
		# chosen_SWA = filter_atoms(f_name+'_SWA.csv', epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [0.2,4], [0.5,6], 1.0, [65,300])
		# chosen_alpha = filter_atoms(f_name+'_alpha.csv', epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [8,11.5], [1.55,np.inf], 1.0, [4,50])
		# chosen_beta = filter_atoms(f_name+'_beta.csv', epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [15,25], [0.4,np.inf], 1.0, [3,50])
		# chosen_theta = filter_atoms(f_name+'_theta.csv', epoch_len, corrupted_epochs, ptspmV, chann_atoms, fs, [4,8], [0.5,np.inf], 1.0, [15,np.inf])

	# df = pd.DataFrame(chosen_SS, columns=['book_id','position','modulus','amplitude','width','frequency','offset','struct_len'])


	# w = 20
	# window = int(w * fs)
	# s = b.signals[channel]
	# sig_len = int(len(s)*20*fs)
	# writer = tags_writer.TagsFileWriter('artifacts.tag')   
	# for idx,i in enumerate(xrange(0, sig_len - window, int(window))):
	#     epoch_id = idx+1
	#     if epoch_id in corrupted_epochs:
	#         # fig = py.figure()
	#         # py.plot(s[epoch_id])  
	#         # print epoch_id, i, np.sum(s[epoch_id] ** 2)
	#         # print epoch_id, np.max(np.abs(np.diff(s[epoch_id]))), np.abs(np.max(s[epoch_id])-np.min(s[epoch_id]))
	#         # raw_input("Press Enter to continue...")
	#         # py.show()
	#         # py.close(fig)     
	#         tag = tag_utils.pack_tag_to_dict(i/fs, i/fs+w, 
	#                                          'artifact',{'energy':energy[idx],'idx':epoch_id})
	#         writer.tag_received(tag)
	# writer.finish_saving(0.0)
