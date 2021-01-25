import numpy as np
import h5py 
import json
from numpy import random
import scipy

## Preprocessing/Import H5 Raw Data Module ##
def integer2volts(intVal, oGain = 760):
    return intVal * 2.0/2**12 / oGain


def Map2OR(xy, theta = -160.0):
    # Flip horizontally
    xy[:,0] *= -1
    # Rotate by theta degrees
    th = theta/180*np.pi;
    return xy @ [[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]]


def GetStimuli(lfpFn, lag_ = None):
    f = h5py.File(lfpFn, "r")  # Open HDF5 file
    archived_metapnd_json = f["archived_metapnd_json"][()]
    archived_metapnd_json = json.loads(archived_metapnd_json)
    stimData = None
    if len(archived_metapnd_json["comments"]) > 0:
        for item in archived_metapnd_json["comments"]:
            if item["comment"]["type"] == "EXPERIMENT_METADATA":
                stimData = np.array(item["comment"]["experiment"]["tapeMetadata"]["target"])

    if lag_ is None:    
        argo_timestep = 1/archived_metapnd_json['sensorState']['sampleRate']
        lag = (archived_metapnd_json['fileAttributes']['primaryStartGpioAssertedAtSeqIdMonotonic'] - archived_metapnd_json['fileAttributes']['primaryFirstRecordedSeqIdMonotonic']) * argo_timestep
    else:
        lag = lag_
    
    if stimData is not None:
        stimData[:,4:6] = stimData[:,4:6] + lag
    
    return stimData


def BuildLFPStruct(lfpFn,
        onDemand = True, 
        stim_ind = 0,
        decimateN = 1, 
        randSubsample = None, 
        gridSample = None,
        seedFlag = True, 
        map2ORFlag = True,
        badChanMask = None,
        threshVal = None):

    f = h5py.File(lfpFn, "r")  # Open HDF5 file
    # print(list(f.keys()))  # Explore the hierarchy
    
    # Load meta
    paramsPP = f["paradsp_pipeline_params"][()]
    paramsPP = json.loads(paramsPP)
    timestep = decimateN/paramsPP['outputSampleRateHz']

    # Get stimuli
    stimData = GetStimuli(lfpFn)

    # Load channel meta
    expJsn = f["export_json"][()]  # Read export_json string
    expJsn = json.loads(expJsn)  # Decode JSON to Python dict
    xyEJ = [(expJsn["xys"][k]["x"], expJsn["xys"][k]["y"]) for k in range(len(expJsn["xys"]))]
    xyEJ = np.array(xyEJ)

    
    # Load another channel meta to do with the bcm test
    bcmJsn = f["bundle_connectivity_mask"][()]  # Read export_json string
    bcmJsn = json.loads(bcmJsn)  # Decode JSON to Python dict
    xyBCM = [(bcmJsn['xys'][k]["x"], bcmJsn['xys'][k]["y"]) for k in range(len(bcmJsn["xys"]))]
    xyBCM = np.array(xyBCM)
    oGain = [(bcmJsn['xys'][k]["oGain"]) for k in range(len(bcmJsn["xys"]))]
    oGain = np.array(oGain)

    # Align the bcm mask to xyEJ
    oEJ = np.lexsort(np.flipud(xyEJ.T))
    oBCM = np.lexsort(np.flipud(xyBCM.T))
    oEJrestore = np.argsort(oEJ)
    oGain = oGain[oBCM[oEJrestore]]
    
    if map2ORFlag:
        xyEJ = Map2OR(xyEJ)
        
    # MASK CHANNELS
    mask = oGain != 0
    
    if badChanMask is not None:
        mask = mask & ~badChanMask

    # oGain = oGain[mask]
    #chanInd = np.where(mask)

    if gridSample is not None:
        grid_mask = GridSubselect(xyEJ,gridSample[0],gridSample[1])
    else:
        grid_mask = np.ones(mask.shape,dtype=bool)
    
    mask = mask & grid_mask

    if threshVal is not None:
        sample_data = integer2volts(f["data"][:,:4096])
        rmsPower = np.sqrt(np.mean(sample_data**2, axis=1))
        print("Mean RMS power is %.2g V..." % np.mean(rmsPower))
        print("Std RMS power is %.2g V..." % np.std(rmsPower))
        rms_mask = rmsPower < threshVal
        print("RMS Rejected Channel: %.2f %%..." % (100.*np.float(sum(~rms_mask)/len(rms_mask))) )
        mask = mask & rms_mask

    if randSubsample is not None:
        if seedFlag:
            random.seed(13)
        msk_ = np.where(mask)[0]
        randMask = np.zeros(mask.shape,dtype=bool)
        randChans = msk_[np.random.permutation(msk_.shape[0])[0:randSubsample]]
        randMask[randChans] = True
        mask = mask & randMask

    if onDemand is False:
        LFPdata = f["data"][mask,0:-1:decimateN]
    else:
        LFPdata = lambda t : f["data"][mask,0:t:decimateN]

    xyEJ = xyEJ[mask,:]
            
    return LFPdata, timestep, xyEJ, mask, stimData


def GridSubselect(xy,n,selection):
    # Divides up the range of a channel map by n*n
    # Selection defines the meshgrid ind.

    xv,yv = np.meshgrid(range(n),range(n))
    xv = xv.reshape(-1,1)
    yv = yv.reshape(-1,1)

    # Get extents +-1 to make boundaries easier
    mx = np.min(xy[:,0]) - 1
    Mx = np.max(xy[:,0]) + 1
    my = np.min(xy[:,1]) - 1
    My = np.max(xy[:,1]) + 1

    in_grid = np.zeros([xy.shape[0],1],dtype=bool)
    
    rx = (Mx-mx)/n
    ry = (My-my)/n
    
    is_in_grid = lambda xy,s: xy[0] > mx+rx*xv[s] and \
                              xy[0] <= mx+rx*(xv[s]+1) and \
                              xy[1] > my+ry*yv[s] and \
                              xy[1] <= my+ry*(yv[s]+1)

    for i,xy_ in enumerate(xy):
        for s in selection:
            if is_in_grid(xy_,s):
                in_grid[i] = True

    return np.squeeze(in_grid)


def subsample_stimuli_matrix(stimData, subSamNum = 3):
    freqs = np.unique(stimData[:,2])
    subFreqs = freqs[::np.ceil(len(freqs)/subSamNum).astype('int')]
    k = [stimData[:,2] == subFreqs[ind] for ind in range(len(subFreqs))]
    mask = np.array(k).any(axis = 0)
    subStimuli = stimData[mask,:]
    
    return subStimuli
