import torch, os, pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_free')
#     memory_available = [int(x.split()[2]) for x in open('gpu_free', 'r').readlines()]
#     gpu = f'cuda:{np.argmax(memory_available)}'
#     if os.path.exists("gpu_free"):
#         os.remove("gpu_free")
#     else:
#           print("The file does not exist") 
#     return gpu

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_free')
    memory_available = [int(x.split()[2]) for x in open('gpu_free', 'r').readlines()]
    gpu = f'cuda:{np.argmin(memory_available)}'
    if os.path.exists("gpu_free"):
        os.remove("gpu_free")
    else:
          print("The file does not exist") 
    return gpu


# explicitly initialize weights for better learning
def initialize_weights(m):
    if isinstance(m, nn.Linear):   #if layer is of Linear
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):   #if layer is of LSTM
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)  #orthogonal is a common way to initialize weights for RNN/LSTM/GRU
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param) #there are really no evidence what works best for convolution, so I just pick one ( He initialization.)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct       = (rounded_preds == y).float() #convert into float for division 
    acc           = correct.sum() / len(correct)
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
    
    
    
import csv, os 
def save_result_csv( row_dic , _path ):
    filename    = _path
    mode        = 'a' if os.path.exists(filename) else 'w'
    with open(f"{filename}", mode) as myfile:
        fileEmpty   = os.stat(filename).st_size == 0
        writer      = csv.DictWriter(myfile, delimiter='|', lineterminator='\n',fieldnames= row_dic.keys())      
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow( row_dic )
        # print(row_dic)
        # print(f"....save file to {filename} success")
        myfile.close()
        
        
def print_cls_var( dict_in ):
    for key, var in vars(dict_in).items():
        print(f'{key} : {var}')