import torch
import torch.nn as nn
import time, copy
from components.helper import epoch_time, binary_accuracy

def train(num_epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device, seq_len_first=False):
    best_valid_loss = float('inf')

    train_losses = []
    train_accs   = []
    valid_losses = []
    valid_accs   = []
    epoch_times   = []
    list_best_epochs = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc    = _train(model, train_loader, optimizer, criterion, device, seq_len_first)
        valid_loss, valid_acc, _ = evaluate(model, val_loader, criterion, device, seq_len_first)
        
        # for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_times.append((epoch_mins, epoch_secs))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            
        list_best_epochs.append(best_epoch)
        
        if (len(list_best_epochs) > 7) and (best_epoch == list_best_epochs[-7]):
            print("Best epoch does not change for 7 epochs >> Break training loop")
            break
            
    test_loss, test_acc, test_predictions = evaluate(best_model, test_loader, criterion, device, seq_len_first)
    print(f"FINAL Best Model from Best Epoch {best_epoch} Test Loss = {test_loss}, Test Acc = {test_acc}")
    # print("="*50)
        
    return train_losses, valid_losses, train_accs, valid_accs, test_loss, test_acc, best_epoch, epoch_times


        
def _train(model, train_loader,  optimizer, criterion, device, seq_len_first=False):

    model.train()
    epoch_train_loss = 0
    epoch_train_acc  = 0

    for i, batch in enumerate(train_loader):
        if (seq_len_first):
            # data shape: (batch, seq len, channel)
            data  = batch[0].to(device).permute(0, 2, 1)    
        else:
            # data shape: (batch, channel, seq len)
            data  = batch[0].to(device)    
        
        # label shape: (batch, 1)
        label = batch[1].to(device) 

        #predict
        output = model(data)  #output shape: (batch, 1)
        
        loss   = criterion(output, label)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #for visualizing
        epoch_train_loss += loss.item()
        acc = binary_accuracy(output, label)
        epoch_train_acc += acc.item()
        
    epoch_train_loss = epoch_train_loss / len(train_loader)
    epoch_train_acc  = epoch_train_acc  / len(train_loader)
    
    return epoch_train_loss, epoch_train_acc


def evaluate(model, val_loader, criterion, device, seq_len_first=False):

    model.eval()
    epoch_val_loss = 0
    epoch_val_acc  = 0
    
    all_predictions = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
            if(seq_len_first):
                # data shape: (batch, seq len, channel)
                data  = batch[0].to(device).permute(0, 2, 1)    
            else:
                # data shape: (batch, channel, seq len)
                data  = batch[0].to(device)  
            
            # label shape: (batch, 1)
            label = batch[1].to(device) 
        
            #predict and cal loss
            output = model(data)
            loss   = criterion(output, label)
            
            rounded_preds = torch.round(torch.sigmoid(output)).long().flatten().tolist()
            all_predictions.extend(rounded_preds)
            
            #for visualizing
            epoch_val_loss += loss.item()
            acc = binary_accuracy(output, label)
            epoch_val_acc += acc.item()
    
    epoch_val_loss =  epoch_val_loss / len(val_loader)
    epoch_val_acc  =  epoch_val_acc  / len(val_loader)
    
    # print(all_predictions)
    # print(len(all_predictions))
    
    return epoch_val_loss, epoch_val_acc, all_predictions

