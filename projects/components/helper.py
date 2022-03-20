import torch
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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