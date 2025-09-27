
# =================================================================================================================================================
# Model classes -----------------------------------------------------------------------------------------------------------------------------------
# =================================================================================================================================================


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
import optuna 
import random
from tqdm import tqdm
import torch.nn as nn                   
import torch.nn.functional as F  
    
# ======================================================================================================

# =================================================================================================================================================
# This script defines neural network models for classifying extreme events 
# and combining multiple data modalities (e.g., tabular + spatial features).
#
# Main components:
#   - ExtremeClassifier: A simple feedforward neural network for binary classification.
#   - ToCombineExtremeClassifier: An MLP that can work standalone as a classifier 
#     or provide hidden features for combination with a CNN.
#   - CombinedModel: A hybrid model that fuses features from both a neural network 
#     and a CNN through fully connected layers for joint classification.
#
# Features:
#   - Handles both NumPy arrays and PyTorch tensors as inputs.
#   - Configurable standalone or combined training modes.
#   - Modular structure for flexible experimentation with NN + CNN combinations.
#
# Dependencies: torch, numpy, xarray, pandas, matplotlib, sklearn, optuna, tqdm, random,
#               functions_improve_CombinedModel
# =================================================================================================================================================

# Simple Neural Network class ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ExtremeClassifier(nn.Module):                        
    def __init__(self,input_dim):                           
        super(ExtremeClassifier,self).__init__()            
        self.fc1 = nn.Linear(input_dim,8)    
                                                 
        self.fc2 = nn.Linear(8,8)               
        self.fc3 = nn.Linear(8,2)              

    def forward(self,x):                          
        if isinstance(x, np.ndarray): 
            x = torch.tensor(x, dtype=torch.float32)  

        x = x.view(-1,self.fc1.in_features)                
        x = F.relu(self.fc1(x))                   
        x = F.relu(self.fc2(x))                          
        x = self.fc3(x)                           

        if not self.training:  
            if not getattr(self, "shap_flag", False):
                x = F.softmax(x, dim=1)        
        return x 

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MLP to combine with the CNN to form the CombinedModel ---------------------------------------------------------------------------------------------------------------------

class ToCombineExtremeClassifier(nn.Module):                         
    def __init__(self,input_dim, train_alone_NN = True, num_classes = 2):                            
        super(ToCombineExtremeClassifier,self).__init__()           

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.BatchNorm1d(15),
            nn.ReLU(),
            nn.Linear(15, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        
        )

        self.train_alone_NN = train_alone_NN

        if self.train_alone_NN:
            self.final_classification = nn.Linear(8, num_classes) 

    def forward(self, x):                         
        if isinstance(x, np.ndarray): 
            x = torch.tensor(x, dtype=torch.float32)  

        x = x.view(-1, self.fc_layers[0].in_features) 
        hidden = self.fc_layers(x) 

        if self.train_alone_NN:
            return  self.final_classification(hidden) #return classification only with NN if training NN alone 
        else:                 
            return hidden

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The Combined Model class ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CombinedModel(nn.Module):

    '''
    Class for the combined model, which takes as input the outputs of the CNN and NN models and combines them through fully connected layers.
    
    nn_model: the NN model
    cnn_model: the CNN model
    nn_hidden_dim: the hidden dimension of the NN model
    cnn_hidden_dim: the hidden dimension of the CNN model
    output_dim: number of output classes (2 for binary classification)
    
    Returns: trained CombinedModel'''

    def __init__(self, nn_model, cnn_model, nn_hidden_dim, cnn_hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.nn_model = nn_model
        self.cnn_model = cnn_model

        combined_hidden_size =  cnn_hidden_dim + nn_hidden_dim 

        self.fc_layers = nn.Sequential(
            nn.Linear(combined_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8,output_dim)
        )
        
    def forward(self, nn_input, cnn_input):
        nn_hidden = self.nn_model(nn_input)
        cnn_hidden = self.cnn_model(cnn_input)

        combined = torch.cat((nn_hidden, cnn_hidden), dim=1)
        output = self.fc_layers(combined)

        return output

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------