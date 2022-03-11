import torch
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def getLoaders(dataset, batch_size):

    # randomly shuffle the indexes, then randomly select 80% of the data
    indices   = torch.randperm(len(dataset)).tolist()
    train_ind = int(0.7 * len(dataset))
    val_ind   = int(0.9 * len(dataset))

    # create subset of dataset
    train_set = torch.utils.data.Subset(dataset, indices[:train_ind])
    val_set   = torch.utils.data.Subset(dataset, indices[train_ind:val_ind])
    test_set  = torch.utils.data.Subset(dataset, indices[val_ind:])

    print(f"Full Dataset size:  {len(dataset)}")
    print(f"Train Dataset size: {len(train_set)}")
    print(f"Valid Dataset size: {len(val_set)}")
    print(f"Test Dataset size:  {len(test_set)}\n")

    # let's create the loader so we can easily loop each batch for training
    params = {"batch_size":16,"shuffle": True,"pin_memory": True}

    train_loader = DataLoader(train_set, **params)
    val_loader   = DataLoader(val_set, **params)
    test_loader  = DataLoader(test_set, **params)
    
    return train_loader, val_loader, test_loader

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