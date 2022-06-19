import mne
from mne_features.feature_extraction import FeatureExtractor
import numpy as np

import logging
from multiprocessing import Pool
from itertools import combinations
import os

def standardize(data: np.ndarray) -> np.ndarray:
    '''data: a numpy array of shape (n_samples, n_features)
    '''
    ori_data = data.copy()
    ori_data = check_float(ori_data)
    ori_data = ori_data.T
    for index, row in enumerate(ori_data):
        mean = row.mean()
        std = row.std()
        row = (row - mean) / std
        ori_data[index] = row
        # print(row)

    return ori_data.T

def check_float(data) -> np.ndarray:
    if(data.dtype == np.int64): data = data.astype(np.float64)
    return data

def convert_to_mne(data: np.ndarray) -> mne.EpochsArray:
    ch_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    ch_types = ['eeg'] * len(ch_names)
    # https://mne.tools/stable/generated/mne.create_info.html
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
    epochs = mne.EpochsArray(data,info, verbose='CRITICAL')
    epochs.set_montage('standard_1020')
    return epochs

def preprocess_interface(data: np.ndarray, variant: str) -> np.ndarray:
    raise ValueError(f"This function should not be called")
    return data

def DE(data: np.ndarray, variant: str) -> np.ndarray:
    bands = [(0,4), (4,8), (8,12), (12,30), (30,64)]
    # [alias_feature_function]__[optional_param]
    params = dict({
        'pow_freq_bands__log':True,
        'pow_freq_bands__normalize':False,
        'pow_freq_bands__freq_bands':bands
    })

    fe = FeatureExtractor(sfreq=128, selected_funcs=['pow_freq_bands'],params=params,n_jobs=1)
    X = fe.fit_transform(X=data)
    return X

def ASYM(data: np.ndarray, variant: str) -> np.ndarray:
    channels = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    left_channels = ['Fp1','F7','F3','T7','P7','C3','P3','O1','AF3','FC5','FC1','CP5','CP1','PO3']
    right_channels = ['Fp2','F8','F4','T8','P8','C4','P4','O2','AF4','FC6','FC2','CP6','CP2','PO4']
    left_channel_indexes = [ channels.index(ch) for ch in left_channels ]
    right_channel_indexes = [ channels.index(ch) for ch in right_channels ]

    frontal_channels = ['FC5','FC1','FC2','FC6','F7','F3','Fz','F4','F8','Fp1','Fp2']
    posterior_channels = ['CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','O1','O2']
    frontal_channel_indexes = [ channels.index(ch) for ch in frontal_channels ]
    posterior_channel_indexes = [ channels.index(ch) for ch in posterior_channels ]

    data_de = DE(data, "")
    PSD_left = data_de[:, left_channel_indexes].copy()
    PSD_right = data_de[:, right_channel_indexes].copy()
    PSD_frontal = data_de[:, frontal_channel_indexes].copy()
    PSD_posterior = data_de[:, posterior_channel_indexes].copy()
    assert variant in ['DASM','RASM','DCAU'], f"The variant {variant} is not supported."

    X = None
    if(variant == 'DASM'):
        X = PSD_left - PSD_right
    elif(variant == 'RASM'):
        X = PSD_left / PSD_right
    elif(variant == 'DCAU'):
        X = PSD_frontal - PSD_posterior

    return X

# connectivity
def pearson_correlation(x,y):
    """ x,y denoted the signal_x and signal_y following the equation """
    cov = np.cov(x, y)
    # print(cov)
    # [[ 8806859.74527069  8007149.0906219 ] ==> [[cov_xx, cov_xy]
    # [ 8007149.0906219  10396797.72458848]]      [cov_yx, cov_yy]]
    cov_xy = cov[0,1] # or cov[1,0]
    cov_xx = cov[0,0]
    cov_yy = cov[1,1]
    corr = cov_xy / ( cov_xx**0.5 * cov_yy**0.5  )
    return corr

def _cal_pcc(p_id, partial_data):
    # print(f"p_id:{p_id} - data to run {partial_data.shape}")
    pcc = []
    for index in range(partial_data.shape[0]):
        pcc_epoch = []
        for comb in combinations(list(range(partial_data.shape[1])), 2):
            pcc_ab = pearson_correlation(partial_data[index, comb[0], :], partial_data[index, comb[1], :]   )
            pcc_epoch.append(pcc_ab)
        pcc_epoch = np.hstack(pcc_epoch)
        pcc.append(pcc_epoch)
    pcc = np.vstack(pcc)
    return pcc

def _parallel(data:np.ndarray, n_jobs:int = 1) -> np.ndarray:
    t_out = 60000
    pool = Pool()
    p_list = []
    ans_list = []
    try:
        indices = np.array_split(np.arange(data.shape[0]), n_jobs)
        for p_id in range(n_jobs):
            p_list.append(pool.apply_async(_cal_pcc, [p_id, data[indices[p_id]] ]))
        for p_id in range(n_jobs):
            ans_list.append( p_list[p_id].get(timeout=t_out) )
    except Exception as e:
        print(e)
    finally:
        print("========= close ========")
        pool.close() 
        pool.terminate()
    ans_list = np.vstack(ans_list)
    return ans_list

def PCC(data, variant: str) -> np.ndarray:
    """ 
    Input: Expect data to have (n_epochs, n_channels, n_samples)
    Output: (n_epochs, n_conn ) => n_conn = n_channels!/(2!(n_channels-2)!)
    """
    assert variant in ['PCC_TIME', 'PCC_FREQ'], f"Variant={variant} is not valid. Variant must be {['PCC_TIME', 'PCC_FREQ']}."
    epochs = convert_to_mne(data)
    epochs = mne.preprocessing.compute_current_source_density(epochs)
    data = epochs.get_data()
    del(epochs)
    if(variant == 'PCC_TIME'):
        pass
    elif(variant == 'PCC_FREQ'):
        data = _calculate_fft(data, 128)
    else:
        # I'm paranoid
        ValueError(f"Variant={variant} is not valid. Variant must be {['PCC_TIME', 'PCC_FREQ']}.")
    ans_list = _parallel(data, n_jobs=os.cpu_count())
    return ans_list
 
def _calculate_fft(signal:np.ndarray, sfreq:int) -> np.ndarray:
    """ signal: can be 1D array of (n_sample,), 2D array of (n_signal, n_sample), or 3D array of (n_epoch, n_signal, n_sample) """
    # the result will be a complex number. We can obtain the magnitude using `absolute`
    magnitude = np.abs(np.fft.fft(signal, n=sfreq, axis=-1))
    # scale the result
    magnitude = magnitude / (sfreq/2)
    # Selecting the range
    magnitude = magnitude.T[:sfreq//2].T
    freq_range= np.fft.fftfreq(sfreq, d=1/sfreq)[:sfreq//2]
    return magnitude