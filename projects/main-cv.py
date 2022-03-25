#!/usr/bin/env python
# coding: utf-8

# # Main 10-Fold Cross-Validation Subject Dependent Split first then Segmentation

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

import time
from components.helper import epoch_time, binary_accuracy, count_parameters, get_par_data, get_segmented_data 
from components.train import train, evaluate, initialize_weights

import os
import pickle
import numpy as np
import time

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_free')
    memory_available = [int(x.split()[2]) for x in open('gpu_free', 'r').readlines()]
    gpu = f'cuda:{np.argmax(memory_available)}'
    if os.path.exists("gpu_free"):
        os.remove("gpu_free")
    else:
          print("The file does not exist") 
    return gpu

device = get_freer_gpu()
print(device)


# In[2]:


debug = False

path = "../data" 
stim = "Arousal"

par_list    = list(range(32))
num_segment = 60
n_split     = 10
sss         = StratifiedShuffleSplit(n_splits = n_split, test_size = 0.25, random_state = 0)

params     = {"batch_size" : 16, "shuffle" : True, "pin_memory" : True}
num_epochs = 50
lr         = 0.0001
model_saved_name = None

if debug:
    par_list    = list(range(2))
    num_epochs = 1
    n_split    = 3
    sss        = StratifiedShuffleSplit(n_splits = n_split, test_size = 0.25, random_state = 0)


# In[3]:


#note that these params are simply obtained from trial and error; I got no theory to back up why I use certain numbers here...
input_dim     = 32 #we got 32 EEG channels
hidden_dim    = 256 #let's define hidden dim as 256
num_layers    = 2  #we gonna have two LSTM layers
output_dim    = 1  #we got 2 classes so we can output only 1 number, 0 for first class and 1 for another class
bidirectional = True  #uses bidirectional LSTM
dropout       = 0.5  #setting dropout to 0.5
seq_len_first = False

## LSTM is the only model that requires seq_len_first = True
# seq_len_first = True
class LSTM(nn.Module):
    '''
    Expected Input Shape: (batch, seq_len, channels)
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
        
    def forward(self, x):
        
        #x = [batch size, seq len, channels]
        out, (hn, cn) = self.lstm(x)
        
        #out = [batch size, seq len, hidden dim * num directions]        
        #hn = [num layers * num directions, batch size, hidden dim]
        #cn = [num layers * num directions, batch size, hidden dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        return self.fc(hn)
    
class Conv1D_LSTM(nn.Module):
    '''
    Expected Input Shape: (batch, channels, seq_len)  <==what conv1d wants
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout):
        super(Conv1D_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=16, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
        
    def forward(self, x):
        # conv1d expects (batch, channels, seq_len)
        # should not try too big a kernel size, which could lead to too much information loss
        x = F.max_pool1d(F.relu(self.norm(self.conv1d(x))), kernel_size=3)
        x = F.max_pool1d(F.relu(self.norm(self.conv1d(x))), kernel_size=3)

        # change to (batch, seq_len, channels) because lstm expects
        x = x.permute(0,2,1)
        
        #x = [batch size, seq len, channels]
        out, (hn, cn) = self.lstm(x)
        
        #out = [batch size, seq len, hidden dim * num directions]        
        #hn = [num layers * num directions, batch size, hidden dim]
        #cn = [num layers * num directions, batch size, hidden dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        return self.fc(hn)

######################################################################################################
    
class Conv1D_LSTM_Attention(nn.Module):
    '''
    Expected Input Shape: (batch, channels, seq_len)  <==what conv1d wants
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout):
        super(Conv1D_LSTM_Attention, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=16, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.unsqueeze(2)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, seq_len, 1]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context = [batch_size, n_hidden * num_directions(=2), seq_len] * [batch_size, seq_len, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.cpu().data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]
    
    def forward(self, x):
        # conv1d expects (batch, channels, seq_len)
        # should not try too many conv1d or there is nothing to attend to, thus I have only left with one layer of conv1d
        x = F.max_pool1d(F.relu(self.norm(self.conv1d(x))), kernel_size=3)

        # change to (batch, seq_len, channels) because lstm expects
        x = x.permute(0,2,1)
        
        #x = [batch size, seq len, channels]
        out, (hn, cn) = self.lstm(x)
        
        #out = [batch size, seq len, hidden dim * num directions]        
        #hn = [num layers * num directions, batch size, hidden dim]
        #cn = [num layers * num directions, batch size, hidden dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        attn_output, attention = self.attention_net(out, hn)

        return self.fc(attn_output)

######################################################################################################
len_reduction = 'mean' # 'sum' or 'last'

class Conv1D_LSTM_SelfAttention(nn.Module):
    '''
    Expected Input Shape: (batch, channels, seq_len)  <==what conv1d wants
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout, len_reduction='mean'):
        super(Conv1D_LSTM_SelfAttention, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=16, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
        self.lin_Q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.lin_K = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.lin_V = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.len_reduction = len_reduction
    
    # lstm_output : [batch_size, seq len, n_hidden * num_directions(=2)]
    def self_attention_net(self, lstm_output):
        q = self.lin_Q(torch.clone(lstm_output))
        k = self.lin_K(torch.clone(lstm_output))
        v = self.lin_V(torch.clone(lstm_output))
        # q : [batch_size, seq_len, n_hidden * num_directions(=2)]
        # k.transpose(1, 2): [batch_size, n_hidden * num_directions(=2), seq_len]
        # attn_w = [batch_size, seq_len, seq_len]
                
        attn_w = torch.matmul(q, k.transpose(1, 2))
        sfmx_attn_w = F.softmax(attn_w, 1)
        
        # context = [batch_size, seq_len, hidden_dim * num_directions(=2)]
        context = torch.matmul(sfmx_attn_w, v)
        
        # by doing some mean/sum, the dimension on the seq len is gone
        if self.len_reduction == "mean":
            return torch.mean(context, dim=1), sfmx_attn_w.cpu().data.numpy()
        elif self.len_reduction == "sum":
            return torch.sum(context, dim=1), sfmx_attn_w.cpu().data.numpy()
        elif self.len_reduction == "last":
            return context[:, -1, :], sfmx_attn_w.cpu().data.numpy()   
        
    def forward(self, x):
        # conv1d expects (batch, channels, seq_len)
        # should not try too big a kernel size, which could lead to too much information loss
        x = F.max_pool1d(F.relu(self.norm(self.conv1d(x))), kernel_size=3)

        # change to (batch, seq_len, channels) because lstm expects
        x = x.permute(0,2,1)
        
        #x = [batch size, seq len, channels]
        out, (hn, cn) = self.lstm(x)
        
        #out = [batch size, seq len, hidden dim * num directions]        
        #hn = [num layers * num directions, batch size, hidden dim]
        #cn = [num layers * num directions, batch size, hidden dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        attn_output, attention = self.self_attention_net(out)

        return self.fc(attn_output)
    
###################################################################################################
n_heads       = 8   #<=======new!
d_k           = (hidden_dim * 2) // n_heads # (256 * 2) // 8
len_reduction = 'mean'  # 'sum' or 'last'

class Conv1D_LSTM_SelfMultiHeadAttention(nn.Module):
    '''
    Expected Input Shape: (batch, channels, seq_len)  <==what conv1d wants
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout, len_reduction='mean'):
        super(Conv1D_LSTM_SelfMultiHeadAttention, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=16, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.softmax       = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(hidden_dim * num_layers, output_dim)
        self.lin_Q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.lin_K = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.lin_V = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.len_reduction = len_reduction
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
    
    # lstm_output : [batch_size, seq len, n_hidden * num_directions(=2)]
    def self_multihead_attention_net(self, lstm_output):
        
        residual, batch_size = lstm_output, lstm_output.size(0) #<---residual added to the last output; batch_size may not be even for the last unit
        
        q = self.lin_Q(torch.clone(lstm_output))
        k = self.lin_K(torch.clone(lstm_output))
        v = self.lin_V(torch.clone(lstm_output))
        # q : [batch_size, seq_len, n_hidden * num_directions(=2)]
        # k.transpose(1, 2): [batch_size, n_hidden * num_directions(=2), seq_len]
        # attn_w = [batch_size, seq_len, seq_len]
        
        #split into heads
        q = q.view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q: [batch_size x n_heads x seq_len x d_k]
        k = k.view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k: [batch_size x n_heads x seq_len x d_k]
        v = v.view(batch_size, -1, n_heads, d_k).transpose(1,2)  # v: [batch_size x n_heads x seq_len x d_k]
        
        # dot production attention
        attn_w = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k) # [batch_size x n_heads x seq_len x seq_len]
                
        sfmx_attn_w = self.softmax(attn_w)
        context = torch.matmul(sfmx_attn_w, v) # [batch_size x n_heads x seq_len x d_k]
        
        # concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_k) # context: [batch_size x seq_len x n_heads * d_k]
        # now context: [batch_size, seq_len, hidden_dim * 2]
        
        # doing skip connection
        # https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
        context = self.layer_norm(residual + context)

        if self.len_reduction == "mean":
            return torch.mean(context, dim=1), sfmx_attn_w.cpu().data.numpy()
        elif self.len_reduction == "sum":
            return torch.sum(context, dim=1), sfmx_attn_w.cpu().data.numpy()
        elif self.len_reduction == "last":
            return context[:, -1, :], sfmx_attn_w.cpu().data.numpy()
        
    def forward(self, x):
        # conv1d expects (batch, channels, seq_len)
        # should not try too big a kernel size, which could lead to too much information loss
        x = F.max_pool1d(F.relu(self.norm(self.conv1d(x))), kernel_size=3)

        # change to (batch, seq_len, channels) because lstm expects
        x = x.permute(0,2,1)
        
        #x = [batch size, seq len, channels]
        out, (hn, cn) = self.lstm(x)
        
        #out = [batch size, seq len, hidden dim * num directions]        
        #hn = [num layers * num directions, batch size, hidden dim]
        #cn = [num layers * num directions, batch size, hidden dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        attn_output, attention = self.self_multihead_attention_net(out)

        return self.fc(attn_output)


# In[4]:


model = LSTM(input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout)
model = model.to(device)  
model.apply(initialize_weights)
print(f'The model {type(model).__name__} has {count_parameters(model):,} trainable parameters')# Train the model

seq_len_first = True

def reset_model():
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout)
    model = model.to(device)  
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion


# In[8]:


result_dict = {}

for par in par_list:
    
    result_dict[par] = {}
    
    # ==== GET ALL DATA OF THIS PAR ====
    all_data, all_label = get_par_data(path, par, stim)

    # ==== GET CV DATA OF THAT PARTICIPANT FOR EACH FOLD ====
    for i_fold, (train_index, test_index) in enumerate(sss.split(all_data, all_label)):
        
        result_dict[par][i_fold] = {}
        
        print(f"Training Par : {par} | Fold {i_fold}")

        X_train, X_test = all_data[train_index]  , all_data[test_index]
        y_train, y_test = all_label[train_index] , all_label[test_index]

        # === PERFORM SEGMENTATION on TRAIN and TEST set === 

        train_data, train_label = get_segmented_data(X_train, y_train, num_segment)
        test_data,  test_label  = get_segmented_data(X_test,  y_test, num_segment)
        del  X_train, X_test, y_train, y_test

        train_dataset = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_label).float())
        test_dataset  = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_label).float())
        del train_data, train_label, test_data, test_label
        
        train_loader = DataLoader(train_dataset, **params)
        val_loader  = DataLoader(test_dataset, **params)

        # === RESET MODEL ===
        model, optimizer, criterion = reset_model()
        
        # === DO TRAINING === 
        train_loss, train_acc, valid_loss, valid_acc = train(num_epochs,
                                                             model,
                                                             train_loader,
                                                             val_loader,
                                                             optimizer,
                                                             criterion, model_saved_name,
                                                             device,
                                                             seq_len_first=seq_len_first)
        result_dict[par][i_fold]['train_loss'] = train_loss
        result_dict[par][i_fold]['train_acc']  = train_acc
        result_dict[par][i_fold]['valid_loss'] = valid_loss
        result_dict[par][i_fold]['valid_acc']  = valid_acc
        
        del model, optimizer, criterion, train_loader, val_loader

    par_train_loss = [result_dict[par][i]['train_loss'][-1] for i in range(n_split)]
    par_train_acc = [result_dict[par][i]['train_acc'][-1]   for i in range(n_split)]
    par_valid_loss = [result_dict[par][i]['valid_loss'][-1] for i in range(n_split)]
    par_valid_acc = [result_dict[par][i]['valid_acc'][-1]   for i in range(n_split)]
    
    print(f"Par {par} AVG train_loss = {np.mean(par_train_loss)}")
    print(f"Par {par} AVG train_acc = {np.mean(par_train_acc)}")    
    print(f"Par {par} AVG valid_loss = {np.mean(par_valid_loss)}")    
    print(f"Par {par} AVG valid_acc = {np.mean(par_valid_acc)}")    
    
with open(f'./models/DEAP_LSTM_{stim}_results.pkl', 'wb') as outp:
    pickle.dump(result_dict, outp, pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:




