import sys
import os
import datetime
from ImportH5 import *
from MNEPreprocessing import *
import TransformClasses
from Significance import *

import time


# Parameters
nChannels_ = 100
threshVal_ = 0.006
dur_ = 120
tmax_ = 0.5
offset_ = 0
freq_band = 'highgamma'
spk_ind = 1

udr_method_0 = "baseline"
rank0 = "none"
udr_method_1 = "svca"
rank1 = 128
udr_method_2 = "ica"
rank2 = 32
udr_method_3 = "carrelation"
rank3 = 1

sdr_method_1 = "vectorizer"
sdr_projection = "None"
sdr_model = "lda"
estimator_ = "oas"
sdr_reg_ = 0.3
reg_ = 0.3
n_components_ = 34

clf_reg_ = 1.0
penalty_ = "l2"
solver_ = "eigen"
shrinkage_ = "auto"
n_bins_ = 6


fname = 'ch' + str(n_Channels_) +  str(udr_method_1) + '_' + 
str(udr_method_2) + str(rank1) + '_' + str(rank2) + sdr_method_1



# Ex-VH hyperparam
test_size_ = 0.2
n_splits_ = 5

sdr_params_ = {item : eval(item) for item in dir() if not item.startswith("_") and item.endswith('_')}


LFPdata, timestep, ChannelCoord, randMask, _ = \
    BuildLFPStruct(lfpfn,
                   True,
                   randSubsample = nChannels_,
                   gridSample = (5, [1,2,6,7,11,12])) # Region of interest selection

raw = BuildMNERawStruct(LFPdata, timestep, ChannelCoord, voltsFlag = False)   

picks_ = mne.pick_types(raw.info, meg='mag')
raw.filter(12, 110, picks = picks_, n_jobs = -1, method = 'fir')

freqs = np.arange(60, 241, 60)
raw.notch_filter(freqs=freqs, picks = picks_, n_jobs = -1)    

if decoding_class == 'phoneme':
	if decoding_type == 'class':
		stimData = sio.loadmat(stim_data_path)
		stimData = stimData['subStim'][()]
		speakers_num = np.unique(stimData[:,3])
		print(speakers_num)
		if spk_ind > len(speakers_num):
		    spk_ind = len(speakers_num)
		spk_val = speakers_num[spk_ind]
		stimDataSub = stimData[stimData[:,3] == spk_val,:]
		raw, event_id = BuildStimuliChannels(stimDataSub, raw, timestep, 'puretone')
	else:

else:
	sent_mat = sio.loadmat(sent_matrix_path)['sentence_matrix']
	raw, event_id = BuildSentenceStimuliChannels(sent_mat, raw, timestep, 'puretone')


# Epoch data
events = mne.find_events(raw, ['puretone'], 'onset', initial_event = True, consecutive = True, min_duration = .01)
picks = mne.pick_types(raw.info, meg='mag')

puretone_epochs = mne.Epochs(raw, events, event_id = event_id, tmin = -0.2, tmax = tmax_, preload = True, baseline = None, picks = picks)

# Setup unsupervised methods
udr_methods_ = [udr_method_0, udr_method_1, udr_method_2, udr_method_3]
udr_ranks_ = [rank0, rank1, rank2, rank3]

## Dimensional Reduction
reduced_puretone_epochs, reduced_basis, mthd_str = \
    TransformClasses.transform_runner(puretone_epochs, methods = udr_methods_, params = udr_ranks_)

# Project data back to full rank channels
udr_proj_epochs = TransformClasses.epochs_projection(reduced_puretone_epochs, reduced_basis)
save_epochs(udr_proj_epochs, fname+'udr_proj_epochs')

save_epochs(reduced_puretone_epochs, fname + 'udr_epochs')

# Decoding step
if decoding_class == 'phoneme':
	if decoding_type = 'classification':
		sdr_method_1 = "erpcov"
		sdr_projection = "tangent_space"
		sdr_model = "logistic"
		sdr_method_ = [sdr_method_1, sdr_projection, sdr_model]
		supervised_dimensional_reduction(reduced_puretone_epochs.copy(), sdr_method_, sdr_params_, fname, binFlag = False)
	else:
		correlation_analysis(reduced_puretone_epochs.copy(), formant_data)
else:
	moses_chang_pipeline(reduced_puretone_epochs.copy(), sdr_params_, fname, nChunk = 4)




