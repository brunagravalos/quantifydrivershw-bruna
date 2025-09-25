# ======================================================================================================
# IMPORT NEEDED PACKAGES
# ======================================================================================================

import torch
import scipy 
import xarray as xr
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import functions_improve_CombinedModel
import optuna 
import random
from tqdm import tqdm
import torch.nn as nn                   
import torch.nn.functional as F  

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ======================================================================================================


# =============================================================================================================================
# Train Models Functions ------------------------------------------------------------------------------------------------------
# =============================================================================================================================
    

# For training the NN model alone

def train_NNmodel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, print_loss, early_stop, patience = 5,print_early_stop=True):

    '''
    Function to train a NN model.
    
    model: NN model to be trained
    combined_train_loader: DataLoader for training data
    combined_val_loader: DataLoader for validation data
    criterion: Loss function
    optimizer: Optimizer
    num_epochs: Number of epochs to train
    plot_loss: Whether to plot the loss curves
    print_loss: Whether to print the loss during training
    early_stop: Whether to use early stopping
    patience: Patience for early stopping
    print_early_stop: Whether to print when early stopping occurs
    
    Returns: losses_train, losses_val
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  

    losses_train = []
    losses_val = []
        
    # training 

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    with tqdm(range(num_epochs), position=0, leave=True) as pbar:
        for _ in pbar:
            total_loss = 0
            train_batches = 0
    
            # For validation
            correct_val_0 = 0    # Correct predictions for class 0
            correct_val_1 = 0    # Correct predictions for class 1
            total_val_0 = 0      # Total samples for class 0
            total_val_1 = 0      # Total samples for class 1
        
    
            for local,regional,label in combined_train_loader:
                local,regional,label = local.to(device), regional.to(device), label.to(device)
                # Only use local and labels 
                optimizer.zero_grad()
                outputs = model(local)
                    
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
                train_batches += 1
    
            losses_train.append(total_loss/train_batches)

            if print_loss:
    
                print(f"Avg Loss: {total_loss/train_batches}")

            # Validation
            model.eval() 
            val_loss = 0.0  
            correct_val = 0  
            total_val = 0 
    
            val_batches = 0
    
            with torch.no_grad():  # No gradient computation 
                for local,regional,label in combined_val_loader:
                    val_local,val_regional,val_labels = local.to(device), regional.to(device), label.to(device)
                    
                    val_outputs = model(val_local)  
                    val_loss += criterion(val_outputs, val_labels).item()  
        
                    _, val_predicted = torch.max(val_outputs, 1)  
                    correct_val += (val_predicted == val_labels).sum().item()  
                    total_val += val_labels.size(0)  
        
                    mask_0 = (val_labels == 0)  
                    correct_val_0 += (val_predicted[mask_0] == val_labels[mask_0]).sum().item()
                    total_val_0 += mask_0.sum().item()
            
                    mask_1 = (val_labels == 1)  
                    correct_val_1 += (val_predicted[mask_1] == val_labels[mask_1]).sum().item()
                    total_val_1 += mask_1.sum().item()
    
                    #update number batches 
                    val_batches += 1
    
            avg_val_loss = val_loss / val_batches
            val_accuracy = correct_val / total_val * 100
            val_0_accuracy = correct_val_0 / total_val_0 * 100 
            val_1_accuracy = correct_val_1 / total_val_1 * 100
    
            losses_val.append(avg_val_loss)
    
            if early_stop:
                if avg_val_loss < best_val_loss: #check validation loss 
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_early_stop:
                            print(f" Early stop at epoch {epoch+1}")
                        break

    # Give back best model of the epoch loop 
    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if plot_loss:

        plt.figure(figsize=(8,4))
        plt.plot(np.array(losses_train),label="train")
        plt.plot(np.array(losses_val),label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss")
        plt.legend()
        plt.show()

    return losses_train, losses_val #, F.softmax(torch.cat(out_probabilitites),dim=1)

# --------------------------------------------------------------------------------------------------------------
# For training the CNN model alone -----------------------------------------------------------------------------

def train_CNNmodel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, print_loss, early_stop, patience = 5,print_early_stop=True):

    '''
    Function to train a CNN model.
    
    model: CNN model to be trained
    combined_train_loader: DataLoader for training data
    combined_val_loader: DataLoader for validation data
    criterion: Loss function
    optimizer: Optimizer
    num_epochs: Number of epochs to train
    plot_loss: Whether to plot the loss curves
    print_loss: Whether to print the loss during training
    early_stop: Whether to use early stopping
    patience: Patience for early stopping
    print_early_stop: Whether to print when early stopping occurs

    Returns: losses_train, losses_val
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  

    losses_train = []
    losses_val = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # training 

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epoch = 0 
    with tqdm(range(num_epochs), position=0, leave=True) as pbar:
        for _ in pbar:
            total_loss = 0
            train_batches = 0
    
            # For validation
            correct_val_0 = 0    # Correct predictions for class 0
            correct_val_1 = 0    # Correct predictions for class 1
            total_val_0 = 0      # Total samples for class 0
            total_val_1 = 0      # Total samples for class 1
        
            for local,regional,label in combined_train_loader:
                local,regional,label = local.to(device), regional.to(device), label.to(device)
        
                optimizer.zero_grad()
                outputs = model(regional)
                
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
                train_batches += 1
    
            losses_train.append(total_loss/train_batches)
    
            if print_loss:
    
                print(f"Avg Loss: {total_loss/train_batches}")
    
            # Validation
            model.eval() 
            val_loss = 0.0  
            correct_val = 0  
            total_val = 0 
    
            val_batches = 0

            with torch.no_grad():  # No gradient computation 
                for local,regional,label in combined_val_loader:
                    val_local,val_regional,val_labels = local.to(device), regional.to(device), label.to(device)
                    
                    val_outputs = model(val_regional)  
                    val_loss += criterion(val_outputs, val_labels).item()  
        
                    _, val_predicted = torch.max(val_outputs, 1)  
                    correct_val += (val_predicted == val_labels).sum().item()  
                    total_val += val_labels.size(0)  
        
                    mask_0 = (val_labels == 0)  
                    correct_val_0 += (val_predicted[mask_0] == val_labels[mask_0]).sum().item()
                    total_val_0 += mask_0.sum().item()
            
                    mask_1 = (val_labels == 1)  
                    correct_val_1 += (val_predicted[mask_1] == val_labels[mask_1]).sum().item()
                    total_val_1 += mask_1.sum().item()
    
                    #update number batches 
                    val_batches += 1
    
            avg_val_loss = val_loss / val_batches
            val_accuracy = correct_val / total_val * 100
            val_0_accuracy = correct_val_0 / total_val_0 * 100 
            val_1_accuracy = correct_val_1 / total_val_1 * 100
    
            losses_val.append(avg_val_loss)
    
            if early_stop:
                epoch += 1
                if avg_val_loss < best_val_loss: #check validation loss 
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_early_stop:
                            print(f" Early stop at epoch {epoch+1}")
                        break

            

    # Give back best model of the epoch loop 
    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if plot_loss:

        plt.figure(figsize=(8,4))
        plt.plot(np.array(losses_train),label="train")
        plt.plot(np.array(losses_val),label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss")
        plt.legend()
        plt.show()

    return losses_train, losses_val, epoch

# -----------------------------------------------------------------------------------------------------------------------------
# For training the combined model ---------------------------------------------------------------------------------------------

def train_CombinedModel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, 
                        print_loss, early_stop, patience = 5,print_early_stop=True, trial=None):

    ''' 
    Function to train a Combined model. Train NN and CNN models simultaneously.
    
    model: Combined model to be trained
    combined_train_loader: DataLoader for training data
    combined_val_loader: DataLoader for validation data
    criterion: Loss function
    optimizer: Optimizer
    num_epochs: Number of epochs to train
    plot_loss: Whether to plot the loss curves
    print_loss: Whether to print the loss during training
    early_stop: Whether to use early stopping
    patience: Patience for early stopping
    print_early_stop: Whether to print when early stopping occurs
    trial: Optuna trial object for hyperparameter optimization (optional)

    Returns: losses_train, losses_val, num_epochs_ran, best_val_loss
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  

    losses_train = []
    losses_val = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # training 

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Track count epochs for early stop 
    num_e = 0
    
    with tqdm(range(num_epochs), position=0, leave=True) as pbar:
        for epoch_num, _ in enumerate(pbar):
            total_loss = 0
            train_batches = 0
    
            # For validation
            correct_val_0 = 0    # Correct predictions for class 0
            correct_val_1 = 0    # Correct predictions for class 1
            total_val_0 = 0      # Total samples for class 0
            total_val_1 = 0      # Total samples for class 1
        
        
            for local,regional,label in combined_train_loader:
                local,regional,label = local.to(device), regional.to(device), label.to(device)
              
                optimizer.zero_grad()
                outputs = model(local, regional)
                
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
                train_batches += 1
    
            losses_train.append(total_loss/train_batches)
    
            if print_loss:
    
                print(f"Avg Loss: {total_loss/train_batches}")
    
            # Validation
            model.eval() 
            val_loss = 0.0  
            correct_val = 0  
            total_val = 0 
    
            val_batches = 0

            with torch.no_grad():  # No gradient computation 
                for local,regional,label in combined_val_loader:
                    val_local,val_regional,val_labels = local.to(device), regional.to(device), label.to(device)
                    
                    val_outputs = model(val_local, val_regional)  
                    val_loss += criterion(val_outputs, val_labels).item()  
        
                    _, val_predicted = torch.max(val_outputs, 1)  
                    correct_val += (val_predicted == val_labels).sum().item()  
                    total_val += val_labels.size(0)  
        
                    mask_0 = (val_labels == 0)  
                    correct_val_0 += (val_predicted[mask_0] == val_labels[mask_0]).sum().item()
                    total_val_0 += mask_0.sum().item()
            
                    mask_1 = (val_labels == 1)  
                    correct_val_1 += (val_predicted[mask_1] == val_labels[mask_1]).sum().item()
                    total_val_1 += mask_1.sum().item()
    
                    #update number batches 
                    val_batches += 1
        
            avg_val_loss = val_loss / val_batches
            val_accuracy = correct_val / total_val * 100
            val_0_accuracy = correct_val_0 / total_val_0 * 100 
            val_1_accuracy = correct_val_1 / total_val_1 * 100
    
            losses_val.append(avg_val_loss)

            if trial:
                # 1. Report the intermediate validation loss to Optuna
                trial.report(avg_val_loss, epoch_num)

                # 2. Check if the trial should be pruned
                if trial.should_prune():
                    # Stop this trial early
                    raise optuna.exceptions.TrialPruned()

            num_e += 1
    
            if early_stop:
                if avg_val_loss < best_val_loss: #check validation loss 
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_early_stop:
                            print(f" Early stop at epoch {num_e+1}")
                        break

    # Give back best model of the epoch loop 
    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if plot_loss:

        plt.figure(figsize=(8,4))
        plt.plot(np.array(losses_train),label="train")
        plt.plot(np.array(losses_val),label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss")
        plt.legend()
        plt.show()

    return losses_train, losses_val, num_e, best_val_loss

# -----------------------------------------------------------------------------------------------------------------------------