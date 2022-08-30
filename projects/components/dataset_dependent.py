from typing import List
import torch
import os, pickle, glob
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from scipy import signal

class Dataset_subjectDependent(torch.utils.data.Dataset):
    '''
    subject-dependent dataset
    '''
    STIMULI_ALL = -1
    STIMULI_VALENCE = 0
    STIMULI_AROUSAL = 1

    def __init__(self, dataset_path: str, lazyload: bool=True):
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
            if(lazyload == False):
                self._load_data(name)

    def get_file_list(self, configuration:str ='dependent') -> List:
        L = list()
        if(configuration == 'dependent'):
            L = list(self.files.keys())
        elif(configuration == 'independent'):
            L = ['all']
        return L

    def get_file_path_list(self):
        return list(self.files.values())
    
    def _load_data(self, name: str):
        if(name not in self.files): 
            raise ValueError(f"The name:{name} are not in {self.files.keys()}")
        path = self.files[name]
        dat = pickle.load(open(path, 'rb'), encoding='latin1')
        # (40, 32, 8064) => (40,32,) remove the first 3 seconds
        
        self.data[name]   = dat['data'][:, :32, 128*3:]
        # (40, 2)
        self.labels[name] = dat['labels'][:,:2]
        
    def get_data_all(self, stimuli):
        # Get all subj in one big datas, labels, groups
        all_datas, all_labels, all_groups = [],[],[]
        for filename in self.get_file_list():
            data, labels, groups = self.get_data(filename, stimuli=stimuli, return_type='numpy')
            # subj 1 will have groups start from 100 101 102 ... 139
            # subj 2 will have groups start from 200 201 202 ... 239
            # ...
            # subj 32 will have groups start from 3200 3201 3202 ... 3239
            groups = int(filename[1:])*100 +  groups
            # print(filename, int(filename[1:])*100 +  groups)
            all_datas.append(data)
            all_labels.append(labels)
            all_groups.append(groups.reshape(-1,1))
        all_datas = np.vstack(all_datas)
        all_labels = np.vstack(all_labels).reshape(-1)
        all_groups = np.vstack(all_groups).reshape(-1)
        return all_datas, all_labels, all_groups

    def get_data(self,name:str, stimuli:int = STIMULI_ALL, sfreq=None , return_type='numpy') -> tuple: 
        if(name == 'all'):
            return self.get_data_all(stimuli)
        if(name not in self.files): 
            raise ValueError(f"The name:{name} are not in {self.files.keys()}")
        # if(return_type not in ['numpy', 'mne']):
        #     raise ValueError(f"Parameter `return_type` must be in ['numpy', 'mne']. You put '{return_type}'")
        # if(return_type == 'mne' and sfreq==None):
        #     raise ValueError(f"When `return_type` is 'mne', `sfreq` must be set")
        if(type(self.data[name]) == type(None) or type(self.labels[name]) == type(None)):
            self._load_data(name)

        data = self.data[name]
        labels = self.labels[name]
        # Select Stimuli
        if(stimuli != self.STIMULI_ALL):
            labels = labels[:, stimuli].reshape(-1,1)
        # Convert Stimuli to 0,1
        labels = self.convert_labels(labels)
        epoch_data, epoch_labels, groups = self._apply_segment(data, labels, return_groups=True)
        if(return_type == 'mne'):
            epoch_data = self._convert_data_to_mne_epochs(epoch_data, sfreq)
        return epoch_data, epoch_labels.reshape(-1), groups
        # if(return_type == 'mne'):
        #     epoch_data = self._convert_data_to_mne_epochs(epoch_data, sfreq)
        return epoch_data, epoch_labels, groups
    
    
    def get_spec_data(self,name:str, stimuli:int = STIMULI_ALL, sfreq=None , return_type='numpy') -> tuple: 
        # if(name not in self.files): 
        #     raise ValueError(f"The name:{name} are not in {self.files.keys()}")
        # if(return_type not in ['numpy', 'mne']):
        #     raise ValueError(f"Parameter `return_type` must be in ['numpy', 'mne']. You put '{return_type}'")
        # if(return_type == 'mne' and sfreq==None):
        #     raise ValueError(f"When `return_type` is 'mne', `sfreq` must be set")
        if(type(self.data[name]) == type(None) or type(self.labels[name]) == type(None)):
            self._load_data(name)
        data   = self.data[name]
        labels = self.labels[name]
        
        # Select Stimuli
        if(stimuli != self.STIMULI_ALL):
            labels = labels[:, stimuli].reshape(-1,1)
            
        # Convert Stimuli to 0,1
        labels = self.convert_labels(labels)
        
        # do Segmentation
        epoch_data, epoch_labels, groups = self._apply_segment(data, labels, return_groups=True)
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

    # def _convert_data_to_mne_epochs(self, data: np.ndarray, sfreq: int):
    #     # convert data to mne.Epochs
    #     ch_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    #     ch_types = ['eeg'] * len(ch_names)
    #     # https://mne.tools/stable/generated/mne.create_info.html
    #     info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    #     epochs = mne.EpochsArray(data,info, verbose='CRITICAL')
    #     epochs.set_montage('standard_1020')
    #     return epochs

    def get_data_torch_dataset(self,name:str, stimuli:int = STIMULI_ALL, train_size=0.75) -> tuple:
        if(name not in self.files): 
            raise ValueError(f"The name:{name} are not in {self.files.keys()}")
        if(type(self.data[name]) == type(None) or type(self.labels[name]) == type(None)):
            self._load_data(name)
        data = self.data[name]
        labels = self.labels[name]
        # Select Stimuli
        if(stimuli != self.STIMULI_ALL):
            labels = labels[:, stimuli].reshape(-1,1)
        # Convert Stimuli to 0,1
        labels = self.convert_labels(labels)
        
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=train_size, shuffle=True, random_state=42)
        # Segmenting
        epoch_data_train, epoch_labels_train = self._apply_segment(data_train, labels_train)
        epoch_data_test, epoch_labels_test = self._apply_segment(data_test, labels_test)
        train_dataset = Dataset(epoch_data_train, epoch_labels_train)
        test_dataset = Dataset(epoch_data_test, epoch_labels_test)
        return train_dataset, test_dataset


    def convert_labels(self, labels, threshold: int=5):
        # print(mean.shape)
        labels = (labels > threshold).astype(float)
        # print((labels > mean).shape)
        return labels

    def set_segment(self, segment_number: int):
        """
            segment_number: from one record, how many smaller record do you want. 
        """
        self.segment = segment_number

    def _apply_segment(self, data, labels, return_groups = False): 
        # data.shape[-1] is time axis.
        if(data.shape[-1] % self.segment != 0):
            raise ValueError(f"The segment={self.segment} causes the unequal window size.\n\t data.shape={data.shape}")

        epoch_data, epoch_labels, groups = [], [], []
        step = data.shape[-1]//self.segment
        for index in np.arange(0, data.shape[-1], step):
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
    
