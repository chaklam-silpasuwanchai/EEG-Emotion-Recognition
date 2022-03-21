import torch, os, pickle
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def getLoaders(dataset, batch_size):
    print(f"Dataset size:  {len(dataset)}")
    # let's create the loader so we can easily loop each batch for training
    params = {"batch_size":batch_size,"shuffle": True,"pin_memory": True}
    loader = DataLoader(dataset, **params)
    
    print(f"Loader size:  {len(loader)}")    
    return loader

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_performance(train, val, name):
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(1, 1, 1)
    ax.plot(train, label = f'train {name}')
    ax.plot(val, label = f'valid {name}')
    plt.legend()
    ax.set_xlabel('updates')
    ax.set_ylabel(name)
    
def get_par_data(path, par, stim):
    
    _, _, filenames = next(os.walk(path))
    filenames = sorted(filenames)
    par_filename = filenames[par]
    print("Par : ", par_filename)

    # ==== GET ALL DATA OF THAT PARTICIPANT ====
    all_data  = []
    all_label = []
    temp = pickle.load(open(os.path.join(path, par_filename), 'rb'), encoding='latin1')
    all_data.append(temp['data'])
    if stim == "Valence":
        all_label.append(temp['labels'][:,:1])  # first index is valence
    elif stim == "Arousal":
        all_label.append(temp['labels'][:,1:2]) # second index is arousal

    # take only the first 32 channels, and take only the first 7680 (not including the 3s baseline)
    all_data  = np.vstack(all_data)[:, :32, :7680]   # shape: (1280, 32, 8064)
    all_label = np.vstack(all_label)            # (1280, 1)  ==> 1280 samples,
    all_label = np.where(all_label >= 5, 1, 0)
    
    return all_data, all_label

def get_segmented_data(data, label, num_segment):
    
    tmp  = data[0, 0, 128:256]
    tmp2 = data[1, 0, 0:128]
    
    data_shape = data.shape
    data = data.reshape(data_shape[0], data_shape[1], num_segment, int(data_shape[2]/num_segment) )
    data = data.transpose(0, 2, 1, 3)
    data = data.reshape(data_shape[0] * num_segment, data_shape[1], -1)
    label = np.repeat(label, num_segment)[:, np.newaxis]  #the dimension 1 is lost after repeat, so need to unsqueeze (896*12, 1)
    
    assert np.array_equal(data[1, 0, :], tmp)
    assert np.array_equal(data[60, 0, :], tmp2)
    
    return data, label