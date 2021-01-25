## Transformations and Analysis
import numpy as np 

import mne
from mne import (io, compute_raw_covariance, read_events, pick_types, Epochs)
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne import Epochs, create_info, events_from_annotations

import scipy.io as sio
import scipy.signal as signal
from scipy.sparse.linalg import svds,eigs
from scipy.stats import pearsonr

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import FastICA

from joblib import dump, load
import dill
import pathos.multiprocessing as mp

import Preprocess

import matplotlib.pyplot as plt

def save_image_to_VH(fig, fname = 'test_fig'):
    output_path = os.getenv('VH_OUTPUTS_DIR')
    filepath = os.path.join(output_path, fname)
    fig.savefig(filepath)


def transform_runner(epochs,methods,params):

    mthd_str = "ch"+str(len(epochs.ch_names))
    firstBasis = True
    master_basis = None
    car_baseline = None

    for m, (mthd,prm) in enumerate(zip(methods,params)):
        
        if mthd is not None or mthd.lower() != 'none':

            if mthd.lower() in {'svds','svd','pca','ica','svca'}:
                # prm is rank
                epochs, basis, s, rank = epochs_dimensional_reduction(epochs,method_=mthd,rank_=prm)
                if m == 0 or firstBasis:
                    master_basis = basis
                    firstBasis = False
                else:
                    master_basis = master_basis @ basis

                mthd_str = mthd_str + "_" + mthd.lower() + str(rank)
            
            elif mthd.lower() == 'meshfilt':
                # Use xy as params
                epochs = Preprocess.epochs_filter(epochs,prm,method_='median',factor_=10)
                mthd_str = mthd_str + "_" + mthd.lower()

            elif mthd.lower() in {'instahz','instantaneous_frequency'}:
                epochs = epochs_instantaneous_frequency(epochs)
                mthd_str = mthd_str + "_instahz"

            elif mthd.lower() == 'spectrogram':
                print("Code not yet implemeted...") #TODO - when needed
                # epochs == epochs_spectrogram(epochs)

            elif mthd.lower() == 'baseline':
                # Calculates car baseline (does not apply it)
                _, car_baseline = epochs_baseline(epochs,apply_car=False)
                mthd_str = mthd_str + "_baseline"

            elif mthd.lower() == 'car':
                epochs, car_baseline = epochs_baseline(epochs,apply_car=True)
                mthd_str = mthd_str + "_car"
            
            elif mthd.lower() == 'carrelation':
                # Takes removes the channels in x most correlated with baseline
                # Removes rank most correlated channels
                if car_baseline is None or prm == 0:
                    print("Skipping Carrelation...")
                else:
                    print("Carrelation parameter: "+str(prm))
                    if prm < 1:
                        # Drop based on top n correlations
                        epochs,basis_mask,r_ = epochs_carrelation(epochs,car_baseline,r_th=prm)
                    else:
                        # Drop based on correlation score
                        epochs,basis_mask,r_ = epochs_carrelation(epochs,car_baseline,drop_n_=prm)

                    mthd_str = mthd_str + "_carrelation"

                    if master_basis is not None:
                        # Delete the dropped channels from the master basis
                        master_basis = master_basis[:,basis_mask]

            elif mthd.lower() == 'drop':
                print("Dropping epochs P2P > " + str(prm) + "...")
                epochs_drop(epochs,reject_=prm)
                mthd_str = mthd_str + "_drop"
            elif mthd.lower() == 'ica_removal':
                print("Removing ICA above 100 Hz")
                epochs, _ = epochs_ica_rejection(epochs, master_basis, cutoff_ = prm)
                mthd_str = mthd_str + "_ica_removal"
            else:
                print("Unrecognized method...")

        else:
            mthd_str = mthd_str + "_none"

    print("Completed " + mthd_str)
    
    return epochs, master_basis, mthd_str


## UNSUPERVISED

# Frequency bases

def instantaneous_frequency(x,fs,axis_=-1):
    # x: data matrix
    # fs: sample frequency
    # Returns: instantaneous frequency and phase
    phase = np.unwrap(np.angle(signal.hilbert(x,axis=axis_)),axis=axis_)
    return np.diff(phase,axis=axis_) / (2.0 * np.pi) * fs, phase


def epochs_instantaneous_frequency(epochs):
    print("Applying instantaneous frequency...")
    fs = 1/np.median(np.diff(epochs.times))
    return mne.EpochsArray(instantaneous_frequency(epochs.get_data(),fs,axis_=-1)[0], \
        epochs.info, epochs.events, epochs.tmin, epochs.event_id)


def epochs_phase(epochs):
    print("Applying phase transform...")
    return mne.EpochsArray(instantaneous_frequency(epochs.get_data(),0,axis_=-1)[1], \
        epochs.info, epochs.events, epochs.tmin, epochs.event_id)


def spectrogram(x,fs_,nperseg_=20,axis_=-1):
    # Returns spectrogram along time dimension
    return signal.spectrogram(x,fs=fs_,nperseg=nperseg_,axis=axis_)


def epochs_spectrogram(epochs,fs_,nperseg_=20):
    # Apply a spectrogram along the time axis and assign the
    # new dimensions into the channel dimension
    pass # TODO


# Decompsitions

def unpack_epochs(epochs):
    # Unpack the epochs for processing
    X = epochs.get_data()
    # Get shape for later
    shp = X.shape
    # Permute so channels is last dim, then flatten epochs and time
    X = np.einsum('ijk->ikj',X)
    X = X.reshape(-1,shp[1]) # reshape with shp[1]: channels
    return X, shp


def odd_even_split(oe,N,L):
    # Create 1D vector of alternating N-islands of 1s and 0s up to L (row major)
    # For use in svca (or other places)
    # oe: 1 for odd 0 for even
    # N: island size
    # L: vector length
    return (np.kron([(x+oe)%2 for x in range(0,-(L//-N))], np.ones([1,N]) )[:L] == 1)[0]


def dimensional_reduction(x,method='svds',rank=256):
    # x: data (samples in rows dim in columns)

    # Update rank_ if decomp produces less columns or rows
    rank = min([rank,x.shape[0]-1,x.shape[1]-1])

    if method.lower() == "svds" or method.lower() == "svd" or method.lower() == "pca":
        print("Running SVDS (rank "+str(rank)+")...")
        x,s,basis = svds(x,k=rank)
        # Reorder svds high to low
        ro = s.argsort()[::-1]
        s = s[ro]
        x = x[:,ro]
        basis = basis[ro,:]
        basis = basis.T
        # x = x @ np.diag(s)
        print("SVDS complete...")
    elif method.lower() == 'ica':
        print("Running ICA (rank "+str(rank)+")...")
        mdl = FastICA(n_components=rank,random_state=0)
        mdl.fit(x)
        basis = mdl.components_.T
        s = (mdl.components_**2).mean(axis=-1)
        s_ind = s.argsort() # ascending
        x = mdl.transform(x)
        basis = basis[:,s_ind]
        s = s[s_ind]
        x = x[:,s_ind]
        print("ICA complete...")
    elif method.lower() == 'svca':
        print("Running SVCA (rank "+str(rank)+")...")
        x,basis,s = svca_decomposition(x,k_=rank,uorv="uv")
        print("SVCA complete...")
    else:
        print("Unknown decomposition method...")

    return x, basis, s, rank


def epochs_dimensional_reduction(epochs, method_='svds', rank_=128):
    # Get data: trials, channels, time
    X = epochs.get_data()
    # Get shape for later
    shp = X.shape

    # Permute so channels is last dim, then flatten epochs and time
    X = np.einsum('ijk->ikj',X)
    X = X.reshape(-1,shp[1]) # reshape with shp[1]: channels

    X, basis, s, rank = dimensional_reduction(X,method=method_,rank=rank_)

    # Reshape the matrix back together
    X = X[:,:rank_].reshape(shp[0],shp[2],-1)
    X = np.einsum('ikj->ijk',X)

    # Add back to epochs
    ch_names_ = [method_+str(n) for n in range(0,X.shape[1])]
    info_ = mne.create_info(ch_names_, epochs.info['sfreq'], 'mag')

    epochs = mne.EpochsArray(X, info_, epochs.events, epochs.tmin, epochs.event_id)

    return epochs, basis, s, rank


def epochs_carrelation(epochs, car_baseline, drop_n_=1, r_th=None, corr_decimation=10):
    # Get data: trials, channels, time
    X = epochs.get_data()
    # Get shape for later
    shp = X.shape

    # Permute so channels is last dim, then flatten epochs and time
    X = np.einsum('ijk->ikj',X)
    X = X.reshape(-1,shp[1]) # reshape with shp[1]: channels

    # Get correlation between the baseline and x
    prsn = lambda x_: pearsonr(np.squeeze(car_baseline)[::corr_decimation],x_[::corr_decimation])[0]

    # Apply correlation
    print("Calculating Pearson Correlation Coefficients...")
    r_ = np.abs(list(map(prsn,X.T)))

    if r_th is None:
        # Order in decreasing correlation
        ord_ = np.argsort(np.abs(r_))[::-1]
        # Crop X
        print("Dropping " + str(drop_n_) + " channels...")
        basis_mask = ord_[drop_n_:]
    else:
        print("Dropping " + str(np.sum(r_ > r_th)) + " channels...")
        basis_mask = np.where(r_ <= r_th)[0]

    X = X[:,basis_mask]

    # Reshape the matrix back together
    X = X.reshape(shp[0],shp[2],-1)
    X = np.einsum('ikj->ijk',X)

    # Add back to epochs
    ch_names_ = ["carrl"+str(n) for n in range(0,X.shape[1])]
    info_ = mne.create_info(ch_names_, epochs.info['sfreq'], 'mag')

    epochs = mne.EpochsArray(X, info_, epochs.events, epochs.tmin, epochs.event_id)

    print("Carrelation Complete...")
    
    return epochs, basis_mask, r_

def epochs_ica_rejection(epochs, master_basis, cutoff = 10):
    timestep = 1/epochs.info['sfreq']
    # Get data: trials, channels, time
    X = epochs.get_data()
    # Get shape for later
    shp = X.shape
    # Permute so channels is last dim, then flatten epochs and time
    X = np.einsum('ijk->ikj',X)
    X = X.reshape(-1,shp[1]) # reshape with shp[1]: channels

    new_order = np.zeros([master_basis.shape[1],1])
    for i in range(master_basis.shape[1]):
        hz_,pxx_ = scipy.signal.welch(X[:,i],fs=1/timestep,nperseg=2**12)
        new_order[i] = np.sum(pxx_[hz_>100])
    new_order = np.squeeze(np.argsort(new_order.T))
    X = X[:,new_order]
    X = X[:, cutoff]
    # Reshape the matrix back together
    X = X.reshape(shp[0],shp[2],-1)
    X = np.einsum('ikj->ijk',X)

    # Add back to epochs
    ch_names_ = ["icacorrected"+str(n) for n in range(0,X.shape[1])]
    info_ = mne.create_info(ch_names_, epochs.info['sfreq'], 'mag')

    epochs = mne.EpochsArray(X, info_, epochs.events, epochs.tmin, epochs.event_id)

    print("ICA Rejection Complete...")
    
    return epochs, new_order



def epochs_drop(epochs,reject_=None):
    if reject_ == 0:
        reject_ = None
    if reject_ is not None:
        reject_ = {'mag':reject_}
    return epochs.drop_bad(reject=reject_)


def epochs_baseline(epochs,apply_car=False):
    print("Calculating (not fitting) CAR baseline...")
    X, shp = unpack_epochs(epochs)
    baseline = X.mean(axis=1,keepdims=True)
    print("Baseline Constructed...")
    if apply_car:
        print("Applying CAR...")
        # Reshape the matrix back together
        X = X.reshape(shp[0],shp[2],-1)
        X = np.einsum('ikj->ijk',X)
        # Add back to epochs
        ch_names_ = ["car"+str(n) for n in range(0,X.shape[1])]
        info_ = mne.create_info(ch_names_, epochs.info['sfreq'], 'mag')
        epochs = mne.EpochsArray(X, info_, epochs.events, epochs.tmin, epochs.event_id)
        print("CAR complete...")

    return epochs, baseline


def epochs_projection(epochs,basis):
    # Input the forward model basis (cascaded filters)
    # Get data: trials, latents, time
    X = epochs.get_data()
    # Get shape for later
    shp = X.shape
    # Permute so latent is last dim, then flatten epochs and time
    X = np.einsum('ijk->ikj',X)
    X = X.reshape(-1,shp[1]) # reshape with shp[1]: channels

    # X*B = latents --> X = latents * pinv(B)
    X = X @ np.linalg.pinv(basis)

    # Reshape the matrix back together
    X = X.reshape(shp[0],shp[2],-1)
    X = np.einsum('ikj->ijk',X)

    # Add back to epochs
    ch_names_ = ["recon_ch"+str(n) for n in range(0,X.shape[1])]
    info_ = mne.create_info(ch_names_, epochs.info['sfreq'], 'mag')

    return mne.EpochsArray(X, info_, epochs.events, epochs.tmin, epochs.event_id)


def filters2patterns(basis):
    return np.linalg.pinv(basis).T


def interlace_uv(u,v,iu,iv):
    # Recombine svca u and v vectors into one master basis.
    # The validity of this operation is not confirmed
    uv = np.zeros([u.shape[0]+v.shape[0],u.shape[1]])
    # Normalisation is applied to adjust for u and v effectively
    # adding two latents together (keep vectors unitary)
    # For equal checkerboard split this would be 0.5 for both
    uv[iu,:] = u * sum(iu)/len(iu) 
    uv[iv,:] = v * sum(iv)/len(iv)
    return uv

def svca_decomposition(x,k_=128,uorv="uv",ch_stride=1):

    # This lambda function does 2D logical indexing (matlab: x(i1,i2))
    # checker = lambda x, i1, i2: x[np.ix_(i1,i2)]

    # Checkerboard split up channels
    trn_ch = odd_even_split(0,ch_stride,x.shape[1])
    tst_ch = odd_even_split(1,ch_stride,x.shape[1])

    # Mean zero data
    x = x - x.mean(axis=0,keepdims=True)

    # Cross covariance
    C = x[:,trn_ch].T @ x[:,tst_ch]

    k_ = min([k_,sum(trn_ch)-1,sum(tst_ch)-1])

    print("SVCA rank update for group split: "+str(k_))

    u,s,vt = svds(C,k=k_)
    ro = s.argsort()[::-1]
    s = s[ro]
    u = u[:,ro]
    vt = vt[ro,:]

    if uorv.lower() == "u":
        basis = u
        x = x[:,trn_ch] @ basis
    elif uorv.lower() == "v":
        basis = vt.T
        x = x[:,tst_ch] @ basis # note svds returns v.T
    elif uorv.lower() == "uv":
        # Interlace u & v together then project all channels
        # Warning: not validated as technique
        basis = interlace_uv(u,vt.T,trn_ch,tst_ch)
        x = x @ basis

    ## Element multiply the sum along rows
    # sn = np.einsum('ij,ij->i',s1,s2)
    # vn = np.einsum('ij->i',s1**2 + s2**2)/2

    return x, basis, s
