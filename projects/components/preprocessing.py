import numpy as np
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

def preprocess_interface(data: np.ndarray, variant: str) -> np.ndarray:
    return data

def DE(data: np.ndarray, variant: str) -> np.ndarray:
    from mne_features.feature_extraction import FeatureExtractor
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