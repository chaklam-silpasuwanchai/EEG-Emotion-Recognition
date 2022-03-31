#!/usr/bin/env python
# coding: utf-8

# # Main 10-Fold Cross-Validation Subject Dependent Split first then Segmentation

# In[31]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from tqdm.notebook import tqdm

from components.models import *
from components.helper import *
from components.dataset_jo import *
from components.train import *

import os
import pickle
import numpy as np
import time

import argparse


# ## Training Configurations

# In[38]:



class Config():
    def __init__(self):
        
        
        
        '''
        LSTM
        Conv1D_LSTM
        Conv1D_LSTM_Attention
        Conv1D_LSTM_SelfAttention
        Conv1D_LSTM_MultiHeadSelfAttention
        '''
        
        
        # set running mode : juypyter or py
        # - jupyter = testing mode
        # - py      = production mode
        parser  = argparse.ArgumentParser()
        parser.add_argument('-a', '--model_name',    help='model_name' , type=str, required=False)
        parser.add_argument('-x', '--stim',          help='stim' ,       type=int, required=False)
        parser.add_argument('-s', '--segment',       help='segment' ,    type=int, required=False)
        parser.add_argument('-l', '--len_reduction', help='len_reduction' , type=str, required=False)
        parser.add_argument('-f', '--isdebug',       help='Set running mode' , type=str, required=False)
        args     = parser.parse_args()

        if args.isdebug == 'yes' or 'json' in args.isdebug :
            print("Jupyter mode")
            model_name    = 'LSTM'
            stim          = 1
            len_reduction = 'mean'  # 'mean'  or 'sum' or 'last'
            segment       = 1 # 1, 3, 5

        else:
            model_name    = str(args.model_name)
            stim          = int(args.stim)
            segment       = int(args.segment)
            len_reduction = str(args.len_reduction)  # 'none' or 'mean' or 'sum' or 'last'
            

        
        
               
        
        ##============================================
        #  !!!!!!!!!!!!     DO NOT EDIT BELOW
        #============================================
        
        
        self.device = 'cpu'

        #========== Training Configurations==========
        self.path = "../data" 
        
        
        # STIMULI_VALENCE = 0
        # STIMULI_AROUSAL = 1       
        self.stim      = stim
        self.stim_name = 'VALENCE' if self.stim else 'AROUSAL'
        self.segment   = segment

        self.params     = {"batch_size" : 16, "shuffle" : True, "pin_memory" : True}
        self.num_epochs = 50
        self.lr         = 0.0001

        # true only if using 'LSTM'
        if model_name == 'LSTM' :
            self.seq_len_first = True
        else :
            self.seq_len_first = False

        self.debug = False
        if self.debug:
            self.num_epochs = 1
            self.n_split    = 3

        #========== Model Configurations==========
        # model list 

        
        
        self.model_name    = model_name   # this should be match with the model class
        self.input_dim     = 32   # we got 32 EEG channels
        self.hidden_dim    = 256  # let's define hidden dim as 256
        self.num_layers    = 2    # we gonna have two LSTM layers
        self.output_dim    = 1    # we got 2 classes so we can output only 1 number, 0 for first class and 1 for another class
        self.bidirectional = True # uses bidirectional LSTM
        self.dropout       = 0.5  # setting dropout to 0.5

        # for self attention
        self.len_reduction = len_reduction

        # for multi head attention
        self.n_heads       = 8
        self.d_k           = (self.hidden_dim * 2) // self.n_heads # (256 * 2) // 8
        
        
        #========== save config ==========
        self.segsplit      = 'split'
        self.output_path   = f'./output/{self.segsplit}_{int(60/self.segment)}s/'
        self.result_csv    = f'{self.output_path}{self.model_name}_result.csv'
        


# In[39]:


config = Config()
print_cls_var( config )


# ## Model Configurations

# In[26]:


def init_model( config ):
    
    if config.model_name == 'LSTM' :
        model = LSTM( config.input_dim, 
                     config.hidden_dim, 
                     config.num_layers, 
                     config.output_dim, 
                     config.bidirectional, 
                     config.dropout)
        
    elif config.model_name == 'Conv1D_LSTM' :
        model = Conv1D_LSTM( config.input_dim, 
                            config.hidden_dim, 
                            config.num_layers, 
                            config.output_dim, 
                            config.bidirectional, 
                            config.dropout
                           )
    elif config.model_name == 'Conv1D_LSTM_Attention' :
        model = Conv1D_LSTM_Attention ( config.input_dim, 
                                       config.hidden_dim, 
                                       config.num_layers, 
                                       config.output_dim, 
                                       config.bidirectional, 
                                       config.dropout
                                      )

    elif config.model_name == 'Conv1D_LSTM_SelfAttention' :
        model = Conv1D_LSTM_SelfAttention( config.input_dim, 
                                  config.hidden_dim, 
                                  config.num_layers, 
                                  config.output_dim, 
                                  config.bidirectional, 
                                  config.dropout, 
                                  config.len_reduction   
                                 )
    elif config.model_name == 'Conv1D_LSTM_MultiHeadSelfAttention' :
        model =Conv1D_LSTM_MultiHeadSelfAttention( config.input_dim, 
                                                  config.hidden_dim, 
                                                  config.num_layers, 
                                                  config.output_dim, 
                                                  config.bidirectional, 
                                                  config.dropout, 
                                                  config.len_reduction,
                                                  config.n_heads,
                                                  config.d_k
                                                 )
    
    
    model = model.to(config.device)  
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.lr) 
    criterion = nn.BCEWithLogitsLoss()
    
    
    return model, optimizer, criterion


# In[27]:


model, _, _ = init_model( config )
print(f'The model {type(model).__name__} has {count_parameters(model):,} trainable parameters')# Train the model


# In[28]:


dataset = Dataset_subjectDependent(config.path)
dataset.set_segment(config.segment)

filenames = dataset.get_file_list()
filenames.sort()
print(filenames)


# In[29]:


# def reset_model():
#     model = LSTM(input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout)
#     model = model.to(device)  
#     model.apply(initialize_weights)
#     optimizer = optim.Adam(model.parameters(), lr=lr) 
#     criterion = nn.BCEWithLogitsLoss()
#     return model, optimizer, criterion

def make_dataloader(X_orig, y_orig, train_idxs, test_idxs, params):
    
    X_train, X_test = X_orig[train_idxs] , X_orig[test_idxs]
    y_train, y_test = y_orig[train_idxs] , y_orig[test_idxs]

    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    test_dataset  = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    del X_train, X_test, y_train, y_test

    train_loader = DataLoader(train_dataset, **params)
    val_loader   = DataLoader(test_dataset, **params)
    
    return train_loader, val_loader


# In[30]:


result_dict = {}

for filename in filenames:
    print("==================================================")
    print("Participant : ", filename )
    result_dict[filename] = {}
    
    
    # get participant dataset
    X, y, groups = dataset.get_data(filename, Dataset_subjectDependent.STIMULI_AROUSAL, return_type='numpy')
    print(filename, X.shape, y.squeeze().shape, groups.shape)

    cv = GroupShuffleSplit(n_splits=10, train_size=0.75, random_state=0)
    
    # get each fold for training and testing model

    for fold, ( train_idxs, test_idxs ) in enumerate( cv.split(X, y.squeeze(), groups)):

        print("---------------------")
        print( "fold : ", fold )
        print(train_idxs.shape, test_idxs.shape, set(groups[train_idxs]).intersection(groups[test_idxs]) )
        

        X_orig, y_orig = X.copy(), y.copy()
        train_loader, val_loader = make_dataloader(X_orig, y_orig, train_idxs, test_idxs, config.params)
        
        # === Init MODEL ===
        model, optimizer, criterion = init_model( config )
        
        # === DO TRAINING === 
        train_loss, valid_loss, train_acc , valid_acc , epoch_times = train(config.num_epochs,
                                                             model,
                                                             train_loader,
                                                             val_loader,
                                                             optimizer,
                                                             criterion,
                                                             config.device,
                                                              config.seq_len_first)
        
        del model, optimizer, criterion, train_loader, val_loader

        # save to csv at specific epoch
        for epoch in range( config.num_epochs ) :
                result_csv_dic               = {}
                result_csv_dic['len_reduction']  =  config.len_reduction
                result_csv_dic['par']        =  filename
                result_csv_dic['stim_name']  =  config.stim_name
                result_csv_dic['fold']       =  fold
                result_csv_dic['epoch']      =  epoch
                result_csv_dic['train_loss'] = train_loss[epoch]
                result_csv_dic['valid_loss'] = valid_loss[epoch]
                result_csv_dic['train_acc']  = train_acc[epoch]
                result_csv_dic['valid_acc']  = valid_acc[epoch]
                result_csv_dic['epoch_time'] = epoch_times[epoch]
                save_result_csv( result_csv_dic, config.result_csv )
                
            
        # ## save dictionary of all output result
        # result_dict[filename][fold]['train_loss'].append(train_loss)
        # result_dict[filename][fold]['train_acc'].append(train_acc)
        # result_dict[filename][fold]['valid_loss'].append(valid_loss)
        # result_dict[filename][fold]['valid_acc'].append(valid_acc)
        # result_dict[filename][fold]['epoch_mins'].append(epoch_mins)
        # result_dict[filename][fold]['epoch_secs'].append(epoch_secs)      
        # with open(f'{config.output_path}{config.model_name}_{config.stim_name}_output_dic', 'wb') as outp:
        #     pickle.dump(result_dict, outp, pickle.HIGHEST_PROTOCOL)
        


# In[ ]:





# In[ ]:




