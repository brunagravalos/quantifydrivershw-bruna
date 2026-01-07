# ====================================================================================================================================================
# Functions to evaluate the model --------------------------------------------------------------------------------------------------------------------
# ====================================================================================================================================================
        
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

# ======================================================================================================
# This script provides evaluation utilities for neural network (NN), convolutional neural network (CNN),
# and combined NN+CNN models in binary classification tasks (e.g., extreme vs. non-extreme events).
#
# Main functionalities:
#   - evaluate_model: Evaluate a standalone model on test data with class-specific accuracies.
#   - gather_ensamble_probabilities: Collect predicted probabilities from an ensemble of models.
#   - evaluate_ensamble: Assess ensemble performance using precomputed probabilities.
#   - evaluate_CombinedModel: Evaluate combined NN+CNN models, optionally returning individual model outputs.
#
# Features:
#   - Computes both overall and per-class accuracies (extreme / non-extreme).
#   - Returns true labels, predictions, and class probabilities.
#   - Supports evaluation of ensembles and models trained independently.
#   - GPU acceleration (CUDA) supported.
#
# Dependencies: torch, numpy, scipy, xarray, pandas, matplotlib, sklearn, optuna, tqdm, random,
#               functions_improve_CombinedModel
# ======================================================================================================


def evaluate_model(model, testloader):

    ''' Function to evaluate the model on the test data.

    model : trained model
    testloader : test Dataloader

    Returns: y_true, y_pred, outputs_prob, extreme_accuracy, non_extreme_accuracy
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  
    correct_extreme = 0  
    correct_non_extreme = 0 
    total_extreme = 0  
    total_non_extreme = 0 

    y_true = []
    y_pred = []
    outputs_prob = []

    with torch.no_grad():  
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) 
            _, predicted = torch.max(outputs, 1)

            # Filter outputs acording to 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            outputs_prob.extend(outputs.cpu())
            
            # Compare predictions to true labels
            for i in range(len(labels)):
                if labels[i] == 0:  
                    total_non_extreme += 1
                    if predicted[i] == 0:  
                        correct_non_extreme += 1
                elif labels[i] == 1:  
                    total_extreme += 1
                    if predicted[i] == 1:  
                        correct_extreme += 1
    
    # Calculate accuracy
    extreme_accuracy = correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0
    
    print(f"Correctly classified extreme days: {correct_extreme}/{total_extreme} ({extreme_accuracy * 100:.2f}%)")
    print(f"Correctly classified non-extreme days: {correct_non_extreme}/{total_non_extreme} ({non_extreme_accuracy * 100:.2f}%)")

    return y_true, y_pred, np.array(outputs_prob), extreme_accuracy*100, non_extreme_accuracy*100

# -------------------------------------------------------------------------------------------------

def gather_ensamble_probabilities(CombinedModel,cnn,nn,test_loader):

    ''' Function to return probabilities of ensambles for the binary classification.
    CoombinedModel: MLP combining cnn and nn
    CNN: convolutional neural netwrok model 
    NN: MLP model 

    Returns 2D array with the probabilitites for class 0 and 1 across samples. 
    '''

    CombinedModel.eval() 
    cnn.eval()
    nn.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs_prob = [] 

    with torch.no_grad(): 
        for local,regional,labels in test_loader:
            local,regional,labels = local.to(device), regional.to(device), labels.to(device)

            outputs = F.softmax(CombinedModel(local, regional),dim=1)

            outputs_prob.extend(outputs.cpu())

    return np.array(outputs_prob)


# -------------------------------------------------------------------------------------------------

def evaluate_ensamble(test_loader, probs_ensamble, print_accuracies=True, batch_size=32):
    """
    test_loader : dataloader for test set
    probs_ensamble : 2D array with the probabilitites for class 0 and 1 across samples.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    correct_extreme = 0
    total_extreme = 0

    correct_non_extreme = 0
    total_non_extreme = 0

    y_true = []
    y_pred = []

    for i, (local, regional, labels) in enumerate(test_loader):
        local, regional, labels = local.to(device), regional.to(device), labels.to(device)

        outputs_ensamble = probs_ensamble[batch_size * i:batch_size * (i + 1)]
        predicted = np.argmax(outputs_ensamble, axis=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted)

        # Compare predictions to true labels
        for label, prediction in zip(labels, predicted):
            if label == 1:  # extreme class
                total_extreme += 1
                correct_extreme += (prediction == label).item()
            elif label == 0:  # non-extreme class
                total_non_extreme += 1
                correct_non_extreme += (prediction == label).item()

    # Compute accuracy
    extreme_accuracy = 100 * correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = 100 * correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0

    if print_accuracies:
        print(f'Extreme class accuracy: {extreme_accuracy:.2f}%')
        print(f'Non-extreme class accuracy: {non_extreme_accuracy:.2f}%')

    return y_true, y_pred, extreme_accuracy, non_extreme_accuracy

# -------------------------------------------------------------------------------------------------

def evaluate_CombinedModel(CombinedModel,cnn,nn,test_loader, print_accuracies=True,train_alone=False):

    '''
    Function to evaluate the combined model.
    
    CombinedModel: MLP combining cnn and nn
    CNN: convolutional neural netwrok model
    NN: MLP model
    print_accuracies: whether to print accuracies or not
    train_alone: whether the nn is trained alone or not (if True, return also individual probabilities of the three models)
    
    Returns: y_true, y_pred, outputs_prob, extreme_accuracy, non_extreme_accuracy'''

    CombinedModel.eval()  
    cnn.eval()
    nn.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    correct_extreme = 0
    total_extreme = 0
    
    correct_non_extreme = 0
    total_non_extreme = 0

    y_true = []
    y_pred = []
    predicted_prob = []
    outputs_prob = []
    outputs_prob_cnn = []
    outputs_prob_nn = []
    outputs_prob_combined = []
    
    
    with torch.no_grad(): 
        for local,regional,labels in test_loader:
            local,regional,labels = local.to(device), regional.to(device), labels.to(device)

            if train_alone:
                outputs_nn = F.softmax(nn(local),dim=1)
                outputs_prob_nn.extend(outputs_nn.cpu())
                
                outputs_cnn = F.softmax(cnn(regional),dim=1)
                outputs_prob_cnn.extend(outputs_cnn.cpu())
                
                outputs_comb = F.softmax(CombinedModel(local,regional),dim=1)
                outputs_prob_combined.extend(outputs_comb.cpu())

                # Final probability combinind the three models 
                outputs_three_models = torch.stack([outputs_nn,outputs_nn,outputs_comb]) 
                
                outputs = (outputs_nn*outputs_cnn + outputs_comb)/2
                
            else:
                outputs = F.softmax(CombinedModel(local, regional),dim=1)

        
            _, predicted = torch.max(outputs, 1)
            predicted_prob.extend(predicted.cpu())

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            outputs_prob.extend(outputs.cpu())
    
            # Compare predictions to true labels
            for label, prediction in zip(labels, predicted):
                if label == 1:  # extreme class
                    total_extreme += 1
                    correct_extreme += (prediction == label).item()
                elif label == 0:  # non-extreme class
                    total_non_extreme += 1
                    correct_non_extreme += (prediction == label).item()

    # Compute accuracy
    extreme_accuracy = 100 * correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = 100 * correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0

    if print_accuracies:
        print(f'Extreme class accuracy: {extreme_accuracy:.2f}%')
        print(f'Non-extreme class accuracy: {non_extreme_accuracy:.2f}%')

    if train_alone:
        outputs_prob_cnn = np.array(outputs_prob_cnn)
        outputs_prob_nn = np.array(outputs_prob_nn)
        outputs_prob_combined = np.array(outputs_prob_combined)
    
        individual_probs = np.stack( (outputs_prob_cnn,outputs_prob_nn,outputs_prob_combined) , axis=0)
        
        return y_true, y_pred, np.array(outputs_prob), extreme_accuracy, non_extreme_accuracy, np.max(individual_probs,axis=2)

    else:

        return y_true, y_pred, np.array(outputs_prob), extreme_accuracy, non_extreme_accuracy

