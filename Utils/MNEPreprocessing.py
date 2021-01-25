## MNE Related Functions - Preprocessing 
import mne
import numpy as np
from sklearn import preprocessing
from plotly.offline import iplot, iplot_mpl
from plotly import tools
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
import matplotlib.pyplot as plt
import scipy.io as sio

def save_image_to_VH(fig, fname = 'test_fig'):
    output_path = os.getenv('VH_OUTPUTS_DIR')
    filepath = os.path.join(output_path, fname)
    fig.savefig(filepath)

def integer2volts(intVal, oGain = 760):
    return intVal * 2.0/2**12 / oGain

def BuildMNERawStruct(LFPdata, timestep, xy, mask=None, voltsFlag=True, carFn = None):
    
    if mask is not None:
        info = mne.create_info([",".join(item) for item in np.round(xy[mask,:]).astype(str)], 1/timestep, 'mag')
    else:
        info = mne.create_info([",".join(item) for item in np.round(xy).astype(str)], 1/timestep, 'mag')

    if callable(LFPdata):
        if carFn is None:
            if voltsFlag:
                data = integer2volts(LFPdata(-1))
            else:
                data = LFPdata(-1)
        else:
            carMat = sio.loadmat(carFn)
            car_baseline = np.squeeze(carMat['baseline'])
            if voltsFlag:
                data = integer2volts(LFPdata(-1)) - car_baseline[:-1]
            else:
                data = LFPdata(-1) - car_baseline[:-1]
    else:
        if voltsFlag:
            data = integer2volts(LFPdata)
        else:
            data = LFPdata
    
    raw = mne.io.RawArray(data, info)
    
    return raw

def apply_mne_filtering(raw, lpf = 70, hpf = 110, notch = np.arange(60, 241, 60)):
    ## Plot original power spectrum
    fig = raw.plot_psd(area_mode='std', show=False, average=True)
    save_image_to_VH(fig, fname = 'before_filtering_psd')
    picks_ = mne.pick_types(raw.info, meg='mag')
    ## BPF Filter
    raw.filter(lpf, hpf, picks = picks_, n_jobs = -1, method = 'fir')
    ## Notch 60 Hz removal Filter
    raw.notch_filter(freqs=notch, picks = picks_, n_jobs = -1)
    ## Plot filtered power spectrum
    fig = raw.plot_psd(area_mode='std', show=False, average=True)
    save_image_to_VH(fig, fname = 'after_filtering_psd')
    
    return raw

def BuildSentenceStimuliChannels(stimData, raw, timestep, stimType = 'puretone', dur = dur_, offset = offset_, chunk_var = chunk_var_):
    
    new_stim_times = np.concatenate(np.array(list(map(lambda ind: np.vstack((np.linspace(stimData[ind,2], stimData[ind,3], chunk_var + 1)[0:chunk_var],np.linspace(stimData[ind,2], stimData[ind,3], chunk_var+1)[1:chunk_var+1])).T, range(stimData.shape[0])))))
    stim_labels = np.concatenate(list(map(lambda ind: np.arange(stimData[ind,0] *chunk_var,(stimData[ind,0] *chunk_var)+chunk_var ), range(255))))-chunk_var-1    
    stim_times = np.ceil(new_stim_times/timestep)
    
    le = preprocessing.LabelEncoder()
    puretones_stim_label = le.fit_transform(stim_labels)+2
    ## Added two as - silence:1, tones: 2
    num_tones = len(np.unique(puretones_stim_label))
    print(num_tones)
    print(np.unique(puretones_stim_label - 1))

    stim_index = np.zeros([1,len(raw.times)])
    L1 = []

    for i in range(num_tones): #one for silence
        L1.append('PT' + str(i+1)) # one for indexing
    event_id = {k:v for k,v in zip(L1,np.arange(1,num_tones+2))}
    print(event_id)

    for q,v in enumerate(stim_times):
        stim_index[0, np.int(v[0])+offset:np.int(v[0]) + dur + offset] = puretones_stim_label[q] - 1
    
    stim_info = mne.create_info([stimType], raw.info['sfreq'], ['stim'])
    stim_channel = mne.io.RawArray(stim_index, stim_info)
    raw.add_channels([stim_channel], force_update_info=True)

    return raw, event_id 

def BuildPhonemeStimuliChannels(stimData, raw, timestep, stimType = 'puretone', t_before = 251, dur = 100, plot=False):
    stim_times = np.ceil(stimData[:,4:6]/timestep)
    le = preprocessing.LabelEncoder()
    puretones_stim_label = le.fit_transform(stimData[:,2])+2
    ## Added two as - silence:1, tones: 2
    num_tones = len(np.unique(stimData[:,2]))
    if stimType is 'silence':
        ## Create event ids for channel information
        event_id = {'silence' : 1, 'tone': 2}
        stim_index = np.ones([1,len(raw.times)])
        for q,v in enumerate(stim_times):
        ## silence vs any tone
            stim_index[0,np.int(v[0]):np.int(v[1])] = 2
            stim_index[0, np.int(v[0])-(t_before - dur):np.int(v[0])-1] = 0
            stim_index[0, np.int(v[1])+1:np.int(v[1])+(t_before - dur)] = 0
    elif stimType is 'puretone':
        stim_index = np.zeros([1,len(raw.times)])
        L1 = []
        
        for i in range(num_tones): #one for silence
            L1.append('PT' + str(i+1)) # one for indexing
        event_id = {k:v for k,v in zip(L1,np.arange(1,num_tones+2))}
        
        for q,v in enumerate(stim_times):
            stim_index[0, np.int(v[0]):np.int(v[1])] = puretones_stim_label[q] - 1
    
    elif stimType is 'pt+silence':
        L1 = []
        for i in range(num_tones+1): #one for silence
            L1.append('PT' + str(i+1)) # one for indexing
        event_id = {k:v for k,v in zip(L1,np.arange(1,num_tones+2))}
        stim_index = np.zeros([1,len(raw.times)])
        for q,v in enumerate(stim_times):
        # silence + tones 
            stim_index[0, np.int(v[0]):np.int(v[1])] = puretones_stim_label[q]
            if q%num_tones== 0:
                stim_index[0, np.int(v[0])-t_before:np.int(v[0])-(t_before - dur)] = 1

    stim_info = mne.create_info([stimType], raw.info['sfreq'], ['stim'])
    stim_channel = mne.io.RawArray(stim_index, stim_info)
    raw.add_channels([stim_channel], force_update_info=True)

    return raw, event_id 


def EpochRawData(raw, event_id, stimChannel,
        preStim = -0.2,
        postStim = 0.2,
        initial_event_ = True,
        min_duration_ = .01,
        plot = False,
        reject_ = None):
    
    events = mne.find_events(raw, stimChannel, 'onset', initial_event = initial_event_, consecutive = True, min_duration = min_duration_)
    
    picks = mne.pick_types(raw.info, meg='mag')
    
    if reject_ is not None:
        reject_dict = {'mag':reject_}
    else:
        reject_dict = None

    epochs = mne.Epochs(raw, events,
        reject = reject_dict,
        event_id = event_id,
        tmin = preStim,
        tmax = postStim,
        preload = True,
        baseline = None,
        picks = picks)
    
    if plot:
    
        fig = mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, show=False)
        # convert plot to plotly
    
        update = dict(layout=dict(showlegend=True), data=[dict(name=e) for e in event_id])
        iplot_mpl(plt.gcf(), update)
    
    return epochs
