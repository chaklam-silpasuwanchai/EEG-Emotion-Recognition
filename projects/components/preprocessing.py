import numpy as np
import os

def standardize(data: np.ndarray) -> np.ndarray:
    '''
        data: a numpy array of shape (n_samples, n_features)
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

def DE(data) -> np.ndarray:
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