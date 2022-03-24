import torch
import os, pickle, glob
import numpy as np
from sklearn.model_selection import train_test_split

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
        files = glob.glob('data/*')
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
            self.files[name] = f
            self.data[name] = None
            self.labels[name] = None
            if(lazyload == False):
                self._load_data(name)

    def get_file_list(self):
        return list(self.files.keys())

    def get_file_path_list(self):
        return list(self.files.values())
    
    def _load_data(self,name: str):
        if(name not in self.files): 
            raise ValueError(f"The name:{name} are not in {self.files.keys()}")
        path = self.files[name]
        dat = pickle.load(open(path, 'rb'), encoding='latin1')
        # (40, 32, 8064) => (40,32,) remove the first 3 seconds
        self.data[name] = dat['data'][:, :32, 128*3:]
        # (40, 2)
        self.labels[name] = dat['labels'][:,:2]

    def get_data_numpy(self,name:str, split=True ,stimuli:int = STIMULI_ALL) -> tuple: 
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
        if(split):
            data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=42)
            # Segmenting
            epoch_data_train, epoch_labels_train = self._apply_segment(data_train, labels_train)
            epoch_data_test, epoch_labels_test = self._apply_segment(data_test, labels_test)
            return epoch_data_train, epoch_data_test, epoch_labels_train, epoch_labels_test
        else:
            epoch_data, epoch_labels = self._apply_segment(data, labels)
            return epoch_data, epoch_labels


    def get_data_torch_dataset(self,name:str, stimuli:int = STIMULI_ALL) -> tuple:
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
        
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=42)
        # Segmenting
        epoch_data_train, epoch_labels_train = self._apply_segment(data_train, labels_train)
        epoch_data_test, epoch_labels_test = self._apply_segment(data_test, labels_test)
        train_dataset = Dataset(epoch_data_train, epoch_labels_train)
        test_dataset = Dataset(epoch_data_test, epoch_labels_test)
        return train_dataset, test_dataset


    def convert_labels(self, labels):
        mean = labels.mean(axis=0)
        # print(mean.shape)
        labels[labels < mean] = 0
        labels[labels >= mean] = 1
        # print((labels > mean).shape)
        return labels

    def set_segment(self, segment_number: int):
        self.segment = segment_number

    def _apply_segment(self, data, labels):
        if(data.shape[-1] % self.segment != 0):
            raise ValueError(f"The segment={self.segment} causes the unequal window size.\n\t data.shape={data.shape}")

        epoch_data, epoch_labels = [], []
        step = data.shape[-1]//self.segment
        for index in np.arange(0,data.shape[-1], step):
            epoch_data.append( data[:,:,index:index+step]  )
            epoch_labels.append( labels )
        epoch_data = np.vstack(epoch_data)
        epoch_labels = np.vstack(epoch_labels)

        return epoch_data, epoch_labels


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