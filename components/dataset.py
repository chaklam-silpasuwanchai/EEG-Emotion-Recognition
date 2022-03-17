import torch
import os
import pickle
import numpy as np
from scipy import signal

# setting seed so that splitting process and training process can be reproduced

class Dataset(torch.utils.data.Dataset):
    '''
    subject-independent dataset
    '''
    def __init__(self, path, stim, split):
        
        torch.manual_seed(1)
        
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
                
        data = np.vstack(all_data)[:, :32, ]   #shape: (1280, 32, 8064) --> take only the first 32 channels
        label = np.vstack(all_label) #(1280, 1)  ==> 1280 samples, 
        
        self.data, self.label = self._split(data, label, split)
        
        shape = self.data.shape
        
        #perform segmentation=====
        segments = 12
        
        self.data = self.data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
        #train data shape: (896, 32, 672, 12)

        self.data = self.data.transpose(0, 3, 1, 2)
        #train data shape: (896, 12, 32, 672)

        self.data = self.data.reshape(shape[0] * segments, shape[1], -1)
        #train data shape: (896*12, 32, 672)
        #==========================
        
        self.label = np.repeat(self.label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
        
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
    
    def _split(self, data, label, split):
        # randomly shuffle the indexes, then randomly select 80% of the data
        indices   = torch.randperm(len(data)).tolist()
        train_ind = int(0.7 * len(data))
        val_ind   = int(0.9 * len(data))

        # create idx
        if split == "train":
            idx = indices[:train_ind] 
        
        elif split == "val":
            idx = indices[train_ind:val_ind] 
        
        elif split == "test":
            idx = indices[val_ind:]

        else:
            raise ValueError("train/val/test")
        
        data = data[idx]
        label = label[idx]
        
        return data, label

# subject-dependent



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
    
    
class SegDataset(torch.utils.data.Dataset):
    
    def __init__(self, all_data, all_label, indices):
        self.data   = all_data[indices]
        self.label  = all_label[indices]
        shape = self.data.shape
        
        #perform segmentation=====
        segments = 12
        
        self.data = self.data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
        #train data shape: (896, 32, 672, 12)

        self.data = self.data.transpose(0, 3, 1, 2)
        #train data shape: (896, 12, 32, 672)

        self.data = self.data.reshape(shape[0] * segments, shape[1], -1)
        #train data shape: (896*12, 32, 672)
        #==========================
        self.label = np.repeat(self.label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
        del all_data, all_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data  = self.data[idx]
        single_label = (self.label[idx] > 5).astype(float)   #convert the scale to either 0 or 1 (to classification problem)
        batch = {
            'data' : torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        return batch
    
class SplitDataset(torch.utils.data.Dataset):
    
    def __init__(self, all_data, all_label, indices):
        self.data   = all_data[indices]
        self.label  = all_label[indices]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data  = self.data[idx]
        single_label = (self.label[idx] > 5).astype(float)   #convert the scale to either 0 or 1 (to classification problem)
        batch = {
            'data' : torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        return batch
    
def get_SplitSegSubjectDependentDatasets(path, stim, par):
    
    # ==== GET FILENAME OF THE PARTICIPANT THAT WE WANT ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    par_filename = filenames[par]
    print(par_filename)
    
    # ==== GET ALL DATA OF THAT PARTICIPANT ====
    all_data  = []
    all_label = []
    temp = pickle.load(open(os.path.join(path, par_filename), 'rb'), encoding='latin1')
    all_data.append(temp['data'])
    if stim == "Valence":
        all_label.append(temp['labels'][:,:1])  # first index is valence
    elif stim == "Arousal":
        all_label.append(temp['labels'][:,1:2]) # second index is arousal
    all_data  = np.vstack(all_data)[:, :32, ]   # shape: (1280, 32, 8064) --> take only the first 32 channels
    all_label = np.vstack(all_label)           # (1280, 1)  ==> 1280 samples, 
    
    # ==== SPLIT DATA OF THAT PARTICIPANT ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]

    # ==== GET SEGMENTED DATASET OF THAT PARTICIPANT ====
    train_set = SegDataset(all_data, all_label, train_idx) 
    val_set   = SegDataset(all_data, all_label, val_idx)
    test_set  = SegDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape)    
    
    return train_set, val_set, test_set

def get_SegSplitSubjectDependentDatasets(path, stim, par):
    
    # ==== GET FILENAME OF THE PARTICIPANT THAT WE WANT ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    par_filename = filenames[par]
    print(par_filename)
    
    # ==== GET ALL DATA OF THAT PARTICIPANT ====
    all_data  = []
    all_label = []
    temp = pickle.load(open(os.path.join(path, par_filename), 'rb'), encoding='latin1')
    all_data.append(temp['data'])
    if stim == "Valence":
        all_label.append(temp['labels'][:,:1])  # first index is valence
    elif stim == "Arousal":
        all_label.append(temp['labels'][:,1:2]) # second index is arousal
    all_data  = np.vstack(all_data)[:, :32, ]   # shape: (1280, 32, 8064) --> take only the first 32 channels
    all_label = np.vstack(all_label)           # (1280, 1)  ==> 1280 samples, 
         
    # ==== DO SEGMENTATION ====
    shape = all_data.shape
    segments = 12
    all_data = all_data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
    #train data shape: (896, 32, 672, 12)
    all_data = all_data.transpose(0, 3, 1, 2)
    #train data shape: (896, 12, 32, 672)
    all_data = all_data.reshape(shape[0] * segments, shape[1], -1)
    #train data shape: (896*12, 32, 672)
    all_label = np.repeat(all_label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
    
    # ==== SPLIT SEGMENTED DATA ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]
    
    # ==== GET SPLIT DATASET ====
    train_set = SplitDataset(all_data, all_label, train_idx) 
    val_set   = SplitDataset(all_data, all_label, val_idx)
    test_set  = SplitDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    
    return train_set, val_set, test_set

class TransferLearningDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, stim, par_filename):
        all_data = []
        all_label = []
        for dat in par_filename:
            temp = pickle.load(open(os.path.join(path,dat), 'rb'), encoding='latin1')
            all_data.append(temp['data'])
            if stim == "Valence":
                all_label.append(temp['labels'][:,:1])   #the first index is valence
            elif stim == "Arousal":
                all_label.append(temp['labels'][:,1:2]) # Arousal  #the second index is arousal
                
        self.data = np.vstack(all_data)[:, :32, ]   #shape: (1280, 32, 8064) --> take only the first 32 channels
        self.label = np.vstack(all_label) #(1280, 1)  ==> 1280 samples, 
        
        shape = self.data.shape
        
        # ==== DO SEGMENTATION ====
        segments = 12
        
        self.data = self.data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
        #train data shape: (896, 32, 672, 12)

        self.data = self.data.transpose(0, 3, 1, 2)
        #train data shape: (896, 12, 32, 672)

        self.data = self.data.reshape(shape[0] * segments, shape[1], -1)
        #train data shape: (896*12, 32, 672)
        #==========================
        
        self.label = np.repeat(self.label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
        
        del all_data, all_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        single_data  = self.data[idx]
        single_label = (self.label[idx] > 5).astype(float)   #convert the scale to either 0 or 1 (to classification problem)
        
        batch = {
            'data' : torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }
        return batch
    
def get_TransferLearningDatasets(path, stim):
    
    # ==== GET PARTICIPANT FILENAME FOR EACH SPLIT ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    torch.manual_seed(1)
    num_par = len(filenames)
    indices   = torch.randperm(num_par).tolist()
    train_ind = int(0.85 * num_par)
    val_ind   = int(0.95 * num_par)
    train_par_idx = indices[ : train_ind] 
    val_par_idx   = indices[train_ind : val_ind] 
    test_par_idx  = indices[val_ind :]
    train_par_filename = [filenames[train_par_id] for train_par_id in train_par_idx]
    val_par_filename   = [filenames[val_par_id] for val_par_id in val_par_idx]
    test_par_filename  = [filenames[test_par_id] for test_par_id in test_par_idx]
    print(train_par_filename)
    print(val_par_filename)
    print(test_par_filename)
    
    # ==== USE FILENAME TO GET SEGMENTED DATASET ====
    train_set = TransferLearningDataset(path, stim, train_par_filename) 
    val_set   = TransferLearningDataset(path, stim, val_par_filename)
    test_set  = TransferLearningDataset(path, stim, test_par_filename)
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    
    return train_set, val_set, test_set

def get_SplitSegPoolDataset(path, stim):
       
    # ==== GET ALL DATA FROM ALL PAR ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    all_data  = []
    all_label = []
    for dat in filenames:
        temp = pickle.load(open(os.path.join(path, dat), 'rb'), encoding='latin1')
        all_data.append(temp['data'])
        if stim == "Valence":
            all_label.append(temp['labels'][:,:1])  # first index is valence
        elif stim == "Arousal":
            all_label.append(temp['labels'][:,1:2]) # second index is arousal
    all_data  = np.vstack(all_data)[:, :32, ]   # shape: (1280, 32, 8064) --> take only the first 32 channels
    all_label = np.vstack(all_label)           # (1280, 1)  ==> 1280 samples, 
        
    # ==== SPLIT DATA AT VIDEO LEVEL ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]
    
    # ==== GET SEGMENTED DATASET ====
    train_set = SegDataset(all_data, all_label, train_idx) 
    val_set   = SegDataset(all_data, all_label, val_idx)
    test_set  = SegDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    
    return train_set, val_set, test_set

def get_SegSplitPoolDatasets(path, stim):
    
    # ==== GET ALL DATA FROM ALL PAR ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    all_data  = []
    all_label = []
    for dat in filenames:
        temp = pickle.load(open(os.path.join(path, dat), 'rb'), encoding='latin1')
        all_data.append(temp['data'])
        if stim == "Valence":
            all_label.append(temp['labels'][:,:1])  # first index is valence
        elif stim == "Arousal":
            all_label.append(temp['labels'][:,1:2]) # second index is arousal
    all_data  = np.vstack(all_data)[:, :32, ]   # shape: (1280, 32, 8064) --> take only the first 32 channels
    all_label = np.vstack(all_label)           # (1280, 1)  ==> 1280 samples, 
         
    # ==== DO SEGMENTATION ====
    shape = all_data.shape
    segments = 12
    all_data = all_data.reshape(shape[0], shape[1], int(shape[2]/segments), segments)
    #train data shape: (896, 32, 672, 12)
    all_data = all_data.transpose(0, 3, 1, 2)
    #train data shape: (896, 12, 32, 672)
    all_data = all_data.reshape(shape[0] * segments, shape[1], -1)
    #train data shape: (896*12, 32, 672)
    all_label = np.repeat(all_label, segments)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
    
    # ==== SPLIT SEGMENTED DATA ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]
    
    # ==== GET SPLIT DATASET ====
    train_set = SplitDataset(all_data, all_label, train_idx) 
    val_set   = SplitDataset(all_data, all_label, val_idx)
    test_set  = SplitDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    return train_set, val_set, test_set

def get_NonSegmentOneParDatasets(path, stim, par, truncate = False):
    
    # ==== GET FILENAME OF THE PARTICIPANT THAT WE WANT ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    par_filename = filenames[par]
    print(par_filename)
    
    # ==== GET ALL DATA OF THAT PARTICIPANT ====
    all_data  = []
    all_label = []
    temp = pickle.load(open(os.path.join(path, par_filename), 'rb'), encoding='latin1')
    all_data.append(temp['data'])
    if stim == "Valence":
        all_label.append(temp['labels'][:,:1])  # first index is valence
    elif stim == "Arousal":
        all_label.append(temp['labels'][:,1:2]) # second index is arousal
    if truncate != False :
        all_data  = np.vstack(all_data)[:, :32, : truncate]
    if truncate == False :
        all_data  = np.vstack(all_data)[:, :32, ]   
    all_label = np.vstack(all_label) 
    all_label = np.vstack(all_label)           # (1280, 1)  ==> 1280 samples, 
    
    # ==== SPLIT DATA ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]
    
    # ==== GET DATASET OF THAT PARTICIPANT ====
    train_set = SplitDataset(all_data, all_label, train_idx) 
    val_set   = SplitDataset(all_data, all_label, val_idx)
    test_set  = SplitDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    
    return train_set, val_set, test_set

def get_NonSegmentAllParDatasets(path, stim, truncate = False):
    
    # ==== GET ALL DATA FROM ALL PAR ====
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    all_data  = []
    all_label = []
    for dat in filenames:
        temp = pickle.load(open(os.path.join(path, dat), 'rb'), encoding='latin1')
        all_data.append(temp['data'])
        if stim == "Valence":
            all_label.append(temp['labels'][:,:1])  # first index is valence
        elif stim == "Arousal":
            all_label.append(temp['labels'][:,1:2]) # second index is arousal
            
    if truncate != False :
        all_data  = np.vstack(all_data)[:, :32, : truncate]
    if truncate == False :
        all_data  = np.vstack(all_data)[:, :32, ]   
    all_label = np.vstack(all_label)          
    
    # ==== SPLIT DATA ====
    len_all_ds = len(all_data)
    torch.manual_seed(1)
    indices   = torch.randperm(len_all_ds).tolist()
    train_ind = int(0.85 * len_all_ds)
    val_ind   = int(0.95 * len_all_ds)
    train_idx = indices[ : train_ind] 
    val_idx   = indices[train_ind : val_ind] 
    test_idx  = indices[val_ind :]
    
    # ==== GET SPLIT DATASET ====
    train_set = SplitDataset(all_data, all_label, train_idx) 
    val_set   = SplitDataset(all_data, all_label, val_idx)
    test_set  = SplitDataset(all_data, all_label, test_idx)
    del all_data, all_label
    
    data_train  = train_set[:]['data']
    label_train = train_set[:]['label']
    data_val    = val_set[:]['data']
    label_val   = val_set[:]['label']
    data_test   = test_set[:]['data']
    label_test  = test_set[:]['label']

    print("Train Data shape : " , data_train.shape)
    print("Train Label shape: " , label_train.shape) 
    print("val Data shape   : " , data_val.shape) 
    print("val Label shape  : " , label_val.shape) 
    print("test Data shape  : " , data_test.shape)  
    print("test Label shape : " , label_test.shape) 
    
    return train_set, val_set, test_set
        