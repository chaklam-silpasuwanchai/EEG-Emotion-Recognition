import torch
import os
import pickle
import numpy as np
from scipy import signal

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path, stim):
        _, _, filenames = next(os.walk(path))
        filenames = sorted(filenames)
        all_data = []
        all_label = []
        for dat in filenames:
            temp = pickle.load(open(os.path.join(path,dat), 'rb'), encoding='latin1')
            all_data.append(temp['data'])
            
            if stim == "Valence":
                all_label.append(temp['labels'][:,:1])   #the first index is valence
            elif stim == "Arousal":
                all_label.append(temp['labels'][:,1:2]) # Arousal  #the second index is arousal
                
        self.data = np.vstack(all_data)[:, :32, ]   #shape: (1280, 32, 8064) --> take only the first 32 channels
        
        shape = self.data.shape
        
        #perform segmentation=====
        segments = 12
        
        self.data = self.data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
        #data shape: (1280, 32, 672, 12)

        self.data = self.data.transpose(0, 3, 1, 2)
        #data shape: (1280, 12, 32, 672)

        self.data = self.data.reshape(shape[0] * segments, shape[1], -1)
        #data shape: (1280*12, 32, 672)
        #==========================
        
        self.label = np.vstack(all_label) #(1280, 1)  ==> 1280 samples, 
        self.label = np.repeat(self.label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (1280*12, 1)
        
        del temp, all_data, all_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data  = self.data[idx]
        single_label = (self.label[idx] > 5).astype(float)   #convert the scale to either 0 or 1 (to classification problem)
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        
        return batch
    
    
class SpecDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, stim, sample_rate, window_size, step_size, segments):
        _, _, filenames = next(os.walk(path))
        filenames = sorted(filenames)
        all_data = []
        all_label = []
        for dat in filenames:
            temp = pickle.load(open(os.path.join(path,dat), 'rb'), encoding='latin1')
            all_data.append(temp['data'])
            
            if stim == "Valence":
                all_label.append(temp['labels'][:,:1])   #the first index is valence
            elif stim == "Arousal":
                all_label.append(temp['labels'][:,1:2]) # Arousal  #the second index is arousal    
        
        self.data = np.vstack(all_data)[:, :32, ]   #shape: (1280, 32, 8064) --> take only the first 32 channels
        
        shape = self.data.shape
        
        #perform segmentation=====          
        self.data = self.data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
        #data shape: (1280, 32, 672, 12)

        self.data = self.data.transpose(0, 3, 1, 2)
        #data shape: (1280, 12, 32, 672)

        self.data = self.data.reshape(shape[0] * segments, shape[1], -1)
        #data shape: (1280*12, 32, 672)
        
        #==========================
        
        #perform spectrogram========
        
        sample = self.data[0, 0, :]
        
        freqs, times, spectrogram = self.log_specgram(sample, sample_rate, window_size, step_size)
        
        all_spec_data = np.zeros((self.data.shape[0], self.data.shape[1], spectrogram.shape[0], spectrogram.shape[1]))
        #loop each trial
        for i, each_trial in enumerate(self.data):
        #     print(each_trial.shape) # (channel, seq len) (e.g., 32, 8064)
            for j, each_trial_channel in enumerate(each_trial):
        #         print(each_trial_channel.shape) # (seq len) (e.g., 8064, ) 
                freqs, times, spectrogram = self.log_specgram(each_trial_channel, sample_rate, window_size, step_size)
                all_spec_data[i, j, :, :] = spectrogram
                
        #============================
        
        self.data = all_spec_data
        self.label = np.vstack(all_label)
        self.label = np.repeat(self.label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (1280*12, 1)

        del temp, all_data, all_label
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_label = (self.label[idx] > 5).astype(float)
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        return batch
    
    def log_specgram(self, sample, sample_rate, window_size=20, step_size=10, eps=1e-10):
        #expect sample shape of (number of samples, )
        #thus if we want to use this scipy.signal, we have to loop each trial and each channel
        freqs, times, spec = signal.spectrogram(sample,
                                        fs=sample_rate,
                                        nperseg=window_size,
                                        noverlap=step_size)
        return freqs, times, 10 * np.log(spec.T.astype(np.float32) + eps)