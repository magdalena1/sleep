flow:
-  select_structures.py
-  compute_params.py 
-  spindle.py  -> calinski-harabasz
*- analyse_erds_spectrum.py
*- create_histograms.py
-  compute_params.py
-  group_patients_info.py


select_structures.py
	- wywołanie:
		run select_structures.py /home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/occ_results/*spindle_occ.csv -t 12.0
	oraz
		run select_structures.py /home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/occ_results/*SWA_occ.csv -t 75.0
	- zapisuje nowe pliki *_occ_sel.csv z wybranymi strukturami (spindle > 12/20 uV, swa > 75 uV)

compute_params.py
	- katalog: /home/mzieleniewska/sleep2/eeg_profiles
	- oblicza parametry i tworzy plik *params.csv dla każdego pacjenta
	- wywołanie:
		./compute_params.py /home/mzieleniewska/Coma_sleep_2016/w_projekcie/montage_ears/data_control/*.b #dla kontroli
		./compute_params.py /home/mzieleniewska/Coma_sleep_2016/w_projekcie/montage_ears/data/*.b 		  #dla pacjentow

spindle.py
	- katalog: /home/mzieleniewska/budzik_analiza2 (git branch analyse-sleep-features)
	- przykładowe wywołanie:
		./spindle.py /home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/occ_results/*<structure>_occ_sel.csv  (SWA lub spindle)
	- w pliku nalezy zdefiniować nazwę pliku wyjściowego!
	- procedura klastrowania (calinski-harabasz)

analyse_erds_spectrum.py
	- katalog: /home/mzieleniewska/budzik_analiza3 (git branch mp-spectrum-night)
	- przykładowe wywołanie:
		./analyse_erds_spectrum.py /home/mzieleniewska/Coma_sleep_2016/w_projekcie/montage_ears/data/*.bin

group_patients_info.py
	- katalog: /home/mzieleniewska/sleep2/eeg_profiles
	- grupuje parametry; tworzy plik classification_parameters.csv	

analyse_sleep_features.py
	- katalog: /home/mzieleniewska/budzik_analiza2 (git branch analyse-sleep-features)
	- przeprowadza klasyfikację na podstawie zadanych parametrów
	
create_histograms.py
	- katalog: /home/mzieleniewska/sleep2/eeg_profiles
	- przykładowe wywołanie:
		./create_histograms.py /home/mzieleniewska/empi/from_hpc/data/smp/patients_99rms_new_reader/occ_results/*occ_sel.csv --bin-width 180





