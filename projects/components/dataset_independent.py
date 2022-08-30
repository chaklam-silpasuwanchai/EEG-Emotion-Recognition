import torch
import os, pickle, glob
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from scipy import signal

class Dataset_subjectIndependent(torch.utils.data.Dataset):
    '''
    subject-dependent dataset
    '''
    STIMULI_ALL = -1
    STIMULI_VALENCE = 0
    STIMULI_AROUSAL = 1

    def __init__(self, dataset_path: str):
        """ 
            dataset_path: str
                The path to the dataset
            lazyload: bool
                If True, the data will be loaded when needed.
                If Flase, the data will be loaded at creation time.
        """

        # ['data/s01.dat',
        # 'data/s02.dat',
        # 'data/s03.dat', ...
        files = glob.glob(f'{dataset_path}/*')
        print(f"Found: {len(files)} files")
        #### Init attribute
        self.files = dict()
        self.data = dict()
        self.labels = dict()
        self.segment = 1

        # Set attribute
        for f in files:
            # s01.dat
            filename = os.path.basename(f)
            # s01      .dat
            name, ext = os.path.splitext(filename)
            # files['s01']: "data/s01.dat"
            self.files[name]  = f
            self.data[name]   = None
            self.labels[name] = None
            self._load_all_data()

    def get_file_list(self):
        return list(self.files.keys())

    def get_file_path_list(self):
        return list(self.files.values())
        
    def _load_all_data(self,):
        self.all_data   = []
        self.all_labels = []
        
        for name in self.files.keys():
            path = self.files[name]
            dat = pickle.load(open(path, 'rb'), encoding='latin1')
            # (40, 32, 8064) => (40,32,) remove the first 3 seconds
            self.data[name]   = dat['data'][:, :32, 128*3:]
            self.labels[name] = dat['labels'][:,:2]
            
            self.all_data.append(dat['data'][:, :32, 128*3:])
            self.all_labels.append(dat['labels'][:,:2])
        
        self.all_data   = np.concatenate(self.all_data, axis = 0)
        self.all_labels = np.concatenate(self.all_labels, axis = 0)
    
    def get_all_data(self, stimuli:int = STIMULI_ALL, sfreq=None , return_type='numpy') -> tuple: 

        all_data   = self.all_data
        all_labels = self.all_labels
        # Select Stimuli
        if(stimuli != self.STIMULI_ALL):
            all_labels = all_labels[:, stimuli].reshape(-1,1)
        # Convert Stimuli to 0,1
        all_labels = self.convert_labels(all_labels)
        epoch_data, epoch_labels, groups = self._apply_segment(all_data, all_labels, return_groups=True)
        return epoch_data, epoch_labels, groups
    
    def get_all_spec_data(self, stimuli:int = STIMULI_ALL, sfreq=None , return_type='numpy') -> tuple: 

        all_data   = self.all_data
        all_labels = self.all_labels
        # Select Stimuli
        if(stimuli != self.STIMULI_ALL):
            all_labels = all_labels[:, stimuli].reshape(-1,1)
        # Convert Stimuli to 0,1
        all_labels = self.convert_labels(all_labels)
        epoch_data, epoch_labels, groups = self._apply_segment(all_data, all_labels, return_groups=True)
        print("eeg data shape : ",  epoch_data.shape )
        print("eeg label shape : ", epoch_labels.shape )
        
        # Make spectrogram
        test_sample = epoch_data[0, 0, :]
        freqs, times, spectrogram = self.log_specgram(test_sample)
        
        all_spec_data = np.zeros((epoch_data.shape[0], epoch_data.shape[1], spectrogram.shape[0], spectrogram.shape[1]))
        # Loop each sample
        for i, each_trial in enumerate(epoch_data):
        #     print(each_trial.shape) # (channel, seq len) (e.g., 32, 8064)
            for j, each_trial_channel in enumerate(each_trial):
        #         print(each_trial_channel.shape) # (seq len) (e.g., 8064, ) 
                freqs, times, spectrogram = self.log_specgram(each_trial_channel)
                all_spec_data[i, j, :, :] = spectrogram
        
        print("Spec data shape : ",  all_spec_data.shape )
        print("Spec label shape : ", epoch_labels.shape )
        
        return all_spec_data, epoch_labels, groups 

    def convert_labels(self, labels, threshold: int=5):
        # print(mean.shape)
        labels = (labels > threshold).astype(float)
        # print((labels > mean).shape)
        return labels

    def set_segment(self, segment_number: int):
        self.segment = segment_number

    def _apply_segment(self, data, labels, return_groups = False):
        if(data.shape[-1] % self.segment != 0):
            raise ValueError(f"The segment={self.segment} causes the unequal window size.\n\t data.shape={data.shape}")

        epoch_data, epoch_labels, groups = [], [], []
        step = data.shape[-1]//self.segment
        for index in np.arange(0,data.shape[-1], step):
            epoch_data.append( data[:,:,index:index+step]  )
            epoch_labels.append( labels )
            groups.append(list(range(data.shape[0])))
        epoch_data = np.vstack(epoch_data)
        epoch_labels = np.vstack(epoch_labels)
        groups = np.hstack(groups)
        if(return_groups):
            return epoch_data, epoch_labels, groups
        else: 
            return epoch_data, epoch_labels
        
    def log_specgram(self, sample):
        
        sample_rate = 128
        
        if self.segment in [1,3,5] :
            window_size = 128
        elif self.segment == 60 :
            window_size = 32
        
        # window_size = int(sample_rate)  #1s
        step_size   = window_size * 0.5 #0.5s
        eps         = 1e-10
        
        #expect sample shape of (number of samples, )
        #thus if we want to use this scipy.signal, we have to loop each trial and each channel
        freqs, times, spec = signal.spectrogram(sample,
                                                fs       = sample_rate,
                                                nperseg  = window_size,
                                                noverlap = step_size)
        return freqs, times, 10 * np.log(spec.T.astype(np.float32) + eps)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data  = self.data[idx]
        single_label = self.labels[idx]
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        
        return batch
    
