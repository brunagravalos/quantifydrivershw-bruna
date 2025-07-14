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

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from tqdm import tqdm

import torch.nn as nn                   # provides classes and functions to create and train neural networks
import torch.nn.functional as F         # provides functions for activation functions, loss functions, and other operations


# Training functions ------------------------------------------------------------------------------------------------------------------------------

 
def train_simple_NN(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """ Function for training the simple Neural Network.
    model : model to be trained 
    train_loader : train Dataloader
    val_loader : validation Dataloader 
    criterion : CrossEntropy for binary classification 
    optimizer : Adam 
    num_epochs : number of epochs for training
    """

    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        
        train_loss = 0.0  # Initialize training loss
        correct_train = 0  # Track number of correct predictions
        total_train = 0  # Track total predictions
    
        # Correct classification of the individual labels 
    
        # For training
        correct_train_0 = 0  # Correct predictions for class 0
        correct_train_1 = 0  # Correct predictions for class 1
        total_train_0 = 0    # Total samples for class 0
        total_train_1 = 0    # Total samples for class 1
        
        # For validation
        correct_val_0 = 0    # Correct predictions for class 0
        correct_val_1 = 0    # Correct predictions for class 1
        total_val_0 = 0      # Total samples for class 0
        total_val_1 = 0      # Total samples for class 1
    
        # During each iteration of the training loop, the train_loader provides a batch of {batch_size} days' worth of data
    
        for inputs, labels in train_loader:  # Iterate over training batches
                                             # inputs : tensor of shape (64, num_features)
                                             #labels : tensor of shape (64,)
            
            optimizer.zero_grad()  # Clear the gradients of all optimized tensors
            outputs = model(inputs)  # Compute the forward pass (predictions)
            loss = criterion(outputs, labels)  # Compute the loss. Calculated by comparing the model's predictions for all {batch_size} days with the true labels (0 or 1) for those days
            loss.backward()  # Compute gradients w.r.t. model parameters
            optimizer.step()  # Update model parameters using the computed gradients 
    
            train_loss += loss.item()  # Accumulate batch loss
    
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct_train += (predicted == labels).sum().item()  # Count correct predictions
            total_train += labels.size(0)  # Update total count
            # Update counters for class 0
            mask_0 = (labels == 0)  # Boolean mask for class 0
            correct_train_0 += (predicted[mask_0] == labels[mask_0]).sum().item()
            total_train_0 += mask_0.sum().item()
        
            # Update counters for class 1
            mask_1 = (labels == 1)  # Boolean mask for class 1
            correct_train_1 += (predicted[mask_1] == labels[mask_1]).sum().item()
            total_train_1 += mask_1.sum().item()
    
        # Compute average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100
        train_0_accuracy = correct_train_0 / total_train_0 * 100
        train_1_accuracy = correct_train_1 / total_train_1 * 100
    
    
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize validation loss
        correct_val = 0  # Track number of correct predictions
        total_val = 0  # Track total predictions
    
        with torch.no_grad():  # Disable gradient computation for validation
            for val_inputs, val_labels in val_loader:  # Iterate over validation batches
                val_outputs = model(val_inputs)  # Compute forward pass
                val_loss += criterion(val_outputs, val_labels).item()  # Compute validation loss
    
                # Compute validation accuracy
                _, val_predicted = torch.max(val_outputs, 1)  # Get class with highest probability
                correct_val += (val_predicted == val_labels).sum().item()  # Count correct predictions
                total_val += val_labels.size(0)  # Update total count
    
                # Update counters for class 0
                mask_0 = (val_labels == 0)  # Boolean mask for class 0
                correct_val_0 += (val_predicted[mask_0] == val_labels[mask_0]).sum().item()
                total_val_0 += mask_0.sum().item()
        
                # Update counters for class 1
                mask_1 = (val_labels == 1)  # Boolean mask for class 1
                correct_val_1 += (val_predicted[mask_1] == val_labels[mask_1]).sum().item()
                total_val_1 += mask_1.sum().item()
    
        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100
        val_0_accuracy = correct_val_0 / total_val_0 * 100 
        val_1_accuracy = correct_val_1 / total_val_1 * 100

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")
    
    
def train_cnn(model, train_loader_era5, val_loader_era5, train_loader_era5land, val_loader_era5land, criterion, optimizer, num_epochs):

    """ 
    model : the CNN model 
    train_loader_era5 : features era5 (time,lat,lon)
    train_loader_era5land : features and labels era5land 
    criterion : cross entropy 
    optimizer : adam 
    """

    model.train()  

    for epoch in range(num_epochs):
        total_loss = 0
        train_loss = 0.0  # Initialize training loss
        correct_train = 0  # Track number of correct predictions
        total_train = 0  # Track total predictions
    
        # Correct classification of the individual labels 
    
        # For training
        correct_train_0 = 0  # Correct predictions for class 0
        correct_train_1 = 0  # Correct predictions for class 1
        total_train_0 = 0    # Total samples for class 0
        total_train_1 = 0    # Total samples for class 1
        
        # For validation
        correct_val_0 = 0    # Correct predictions for class 0
        correct_val_1 = 0    # Correct predictions for class 1
        total_val_0 = 0      # Total samples for class 0
        total_val_1 = 0      # Total samples for class 1

        for inputs, (nn_inputs, labels) in zip(train_loader_era5, train_loader_era5land): #use era5 features and era5land labels 

            optimizer.zero_grad()  # Clear the gradients of all optimized tensors
            outputs = model(inputs)  # Compute the forward pass (predictions)
            loss = criterion(outputs, labels)  # Compute the loss. Calculated by comparing the model's predictions for all {batch_size} days with the true labels (0 or 1) for those days
            loss.backward()  # Compute gradients w.r.t. model parameters
            optimizer.step()  # Update model parameters using the computed gradients 
    
            train_loss += loss.item()  # Accumulate batch loss

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct_train += (predicted == labels).sum().item()  # Count correct predictions
            total_train += labels.size(0)  # Update total count
            # Update counters for class 0
            mask_0 = (labels == 0)  # Boolean mask for class 0
            correct_train_0 += (predicted[mask_0] == labels[mask_0]).sum().item()
            total_train_0 += mask_0.sum().item()
        
            # Update counters for class 1
            mask_1 = (labels == 1)  # Boolean mask for class 1
            correct_train_1 += (predicted[mask_1] == labels[mask_1]).sum().item()
            total_train_1 += mask_1.sum().item()
    
        # Compute average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader_era5)
        train_accuracy = correct_train / total_train * 100
        train_0_accuracy = correct_train_0 / total_train_0 * 100
        train_1_accuracy = correct_train_1 / total_train_1 * 100

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize validation loss
        correct_val = 0  # Track number of correct predictions
        total_val = 0  # Track total predictions
    
        with torch.no_grad():  # Disable gradient computation for validation
            for val_inputs, (nn_val_inputs, val_labels) in zip(val_loader_era5, val_loader_era5land):  # Iterate over validation batches. Uses features era5 and labels era5land 
                val_outputs = model(val_inputs)  # Compute forward pass
                val_loss += criterion(val_outputs, val_labels).item()  # Compute validation loss
    
                # Compute validation accuracy
                _, val_predicted = torch.max(val_outputs, 1)  # Get class with highest probability
                correct_val += (val_predicted == val_labels).sum().item()  # Count correct predictions
                total_val += val_labels.size(0)  # Update total count
    
                # Update counters for class 0
                mask_0 = (val_labels == 0)  # Boolean mask for class 0
                correct_val_0 += (val_predicted[mask_0] == val_labels[mask_0]).sum().item()
                total_val_0 += mask_0.sum().item()
        
                # Update counters for class 1
                mask_1 = (val_labels == 1)  # Boolean mask for class 1
                correct_val_1 += (val_predicted[mask_1] == val_labels[mask_1]).sum().item()
                total_val_1 += mask_1.sum().item()
    
        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader_era5)
        val_accuracy = correct_val / total_val * 100
        val_0_accuracy = correct_val_0 / total_val_0 * 100 
        val_1_accuracy = correct_val_1 / total_val_1 * 100

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")





# Function to evaluate the model ---------------------------------------------------------------------------------------------------------------------
        
def evaluate_model(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
    correct_extreme = 0  # Count of correctly classified extreme days
    correct_non_extreme = 0  # Count of correctly classified non-extreme days
    total_extreme = 0  # Total number of extreme days in the test data
    total_non_extreme = 0  # Total number of non-extreme days in the test data

    y_true = []
    y_pred = []
    outputs_prob = []

    
    
    with torch.no_grad():  # Disable gradient tracking during evaluation (saves memory and computations)
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs) #raw logits 
            _, predicted = torch.max(outputs, 1)

            # Filter outputs acording to 
            
            # Predicted class (0 for non-extreme, 1 for extreme)
            #_, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            outputs_prob.extend(outputs.cpu())
            
            # Compare predictions to true labels
            for i in range(len(labels)):
                if labels[i] == 0:  # Non-extreme day
                    total_non_extreme += 1
                    if predicted[i] == 0:  # Correct classification
                        correct_non_extreme += 1
                elif labels[i] == 1:  # Extreme day
                    total_extreme += 1
                    if predicted[i] == 1:  # Correct classification
                        correct_extreme += 1
    
    # Calculate accuracy
    extreme_accuracy = correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0
    
    print(f"Correctly classified extreme days: {correct_extreme}/{total_extreme} ({extreme_accuracy * 100:.2f}%)")
    print(f"Correctly classified non-extreme days: {correct_non_extreme}/{total_non_extreme} ({non_extreme_accuracy * 100:.2f}%)")

    return y_true, y_pred, np.array(outputs_prob), extreme_accuracy*100, non_extreme_accuracy*100


def evaluate_model_confidance(model, testloader,percentage):

    model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
    correct_extreme = 0  # Count of correctly classified extreme days
    correct_non_extreme = 0  # Count of correctly classified non-extreme days
    total_extreme = 0  # Total number of extreme days in the test data
    total_non_extreme = 0  # Total number of non-extreme days in the test data
    
    y_true = []
    y_pred = []
    outputs_prob = []
    
    with torch.no_grad():  # Disable gradient tracking during evaluation (saves memory and computations)
        for inputs, labels in testloader:
            # Forward pass
            outputs = model(inputs) #result is probabilitites because model includes softmax in case we are in evaluation phase (model.eval())
    
            y_true.extend(labels.cpu().numpy())
            outputs_prob.extend(outputs.cpu())
    
        outputs_prob = np.array(outputs_prob)  
        y_true = np.array(y_true)
    
        mask = np.zeros_like(outputs_prob, dtype=bool)
    
        for j in range(outputs_prob.shape[1]):
            threshold = np.percentile(outputs[:, j], 100-percentage)
            
            mask[:, j] = outputs_prob[:, j] >= threshold
                
        mask_2d = mask.copy()
        mask = mask.any(axis=1)
        
        outputs_filtered = outputs_prob[mask] #filter out probabilitites with the mask 
    
        mask_1D = mask_2d[:,0] | mask_2d[:,1]
    
        # Filter outputs acording to % confidence 
        
        # Predicted class (0 for non-extreme, 1 for extreme)
        predicted = np.argmax(outputs_filtered, axis=1)
    
        y_true_filtered = y_true[mask_1D]
            
        y_pred = predicted
            
        # Compare predictions to true labels
        for i in range(len(y_true_filtered)):
            if y_true_filtered[i] == 0:  # Non-extreme day
                total_non_extreme += 1
                if predicted[i] == 0:  # Correct classification
                    correct_non_extreme += 1
            elif y_true_filtered[i] == 1:  # Extreme day
                total_extreme += 1
                if predicted[i] == 1:  # Correct classification
                    correct_extreme += 1
    
    # Calculate accuracy
    extreme_accuracy = correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0
    
    #print(f"Correctly classified extreme days: {correct_extreme}/{total_extreme} ({extreme_accuracy * 100:.2f}%)")
    #print(f"Correctly classified non-extreme days: {correct_non_extreme}/{total_non_extreme} ({non_extreme_accuracy * 100:.2f}%)")

    return y_true_filtered, y_pred, outputs_filtered, extreme_accuracy*100, non_extreme_accuracy*100


def gather_ensamble_probabilities(CombinedModel,cnn,nn,test_loader):

    ''' Function to return probabilities of ensambles for the binary classification.
    CoombinedModel: MLP combining cnn and nn
    CNN: convolutional neural netwrok model 
    NN: MLP model 

    Returns 2D array with the probabilitites for class 0 and 1 across samples. 
    '''

    CombinedModel.eval()  # set model to evaluation mode
    cnn.eval()
    nn.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs_prob = [] # store probabilities from output 

    with torch.no_grad():  # no gradient needed
        for local,regional,labels in test_loader:
            local,regional,labels = local.to(device), regional.to(device), labels.to(device)

            outputs = F.softmax(CombinedModel(local, regional),dim=1)

            outputs_prob.extend(outputs.cpu())

    return np.array(outputs_prob)


def evaluate_ensamble_EOFs(model,test_loader,probs_ensamble, std_ensamble, print_accuracies=True):

    model.eval()

    batch_size = len(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    correct_extreme = 0
    total_extreme = 0
    
    correct_non_extreme = 0
    total_non_extreme = 0

    y_true = []
    y_pred = []

    uncertainty_extreme = []
    uncertainty_nonextreme = []

    with torch.no_grad():  # no gradient needed
        for i,(inputs,labels) in enumerate(test_loader):
            inputs,labels = inputs.to(device), labels.to(device)


            outputs_ensamble = probs_ensamble[batch_size*i:batch_size*(i+1)]
            std = std_ensamble[batch_size*i:batch_size*(i+1)]
            predicted = np.argmax(outputs_ensamble, axis=1)
            error = std[np.arange(std.shape[0]), predicted]

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)


            # Compare predictions to true labels
            for label, prediction,err in zip(labels, predicted,error):
                if label == 1:  # extreme class
                    total_extreme += 1
                    correct_extreme += (prediction == label).item()
                    if prediction == label:
                        uncertainty_extreme.append(err)
                elif label == 0:  # non-extreme class
                    total_non_extreme += 1
                    correct_non_extreme += (prediction == label).item()
                    if prediction == label:
                        uncertainty_nonextreme.append(err)

    uncertainty_extreme = np.sum(np.array(uncertainty_extreme)) / total_extreme
    uncertainty_nonextreme = np.sum(np.array(uncertainty_nonextreme)) / total_non_extreme

    
    # Compute accuracy
    extreme_accuracy = 100 * correct_extreme / total_extreme if total_extreme > 0 else 0
    non_extreme_accuracy = 100 * correct_non_extreme / total_non_extreme if total_non_extreme > 0 else 0

    if print_accuracies:
        print(f'Extreme class accuracy: ({extreme_accuracy:.3f} + {uncertainty_extreme:.3f})%')
        print(f'Non-extreme class accuracy: {non_extreme_accuracy:.3f} + {uncertainty_nonextreme:.3f})%')

    return y_true, y_pred, extreme_accuracy, non_extreme_accuracy



def evaluate_ensamble(CombinedModel,cnn,nn,test_loader,probs_ensamble,print_accuracies=True,batch_size=32):

    CombinedModel.eval()  # set model to evaluation mode
    cnn.eval()
    nn.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    correct_extreme = 0
    total_extreme = 0
    
    correct_non_extreme = 0
    total_non_extreme = 0

    y_true = []
    y_pred = []

    with torch.no_grad():  # no gradient needed
        for i,(local,regional,labels) in enumerate(test_loader):
            local,regional,labels = local.to(device), regional.to(device), labels.to(device)


            outputs_ensamble = probs_ensamble[batch_size*i:batch_size*(i+1)]
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




def evaluate_CombinedModel(CombinedModel,cnn,nn,test_loader, print_accuracies=True,train_alone=False):

    CombinedModel.eval()  # set model to evaluation mode
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
    
    
    with torch.no_grad():  # no gradient needed
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
                outputs_three_models = torch.stack([outputs_nn,outputs_nn,outputs_comb]) #stack the three probs and apply softmax to obtain the final combined probability 
                
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


def evaluate_CNN(model,test_loader,print_accuracies=True):

    model.eval()

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
    
    
    with torch.no_grad():  # no gradient needed
        for local,regional,labels in test_loader:
            local,regional,labels = local.to(device), regional.to(device), labels.to(device)

            outputs = F.softmax(model(regional),dim=1)

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

    return y_true, y_pred, np.array(outputs_prob), extreme_accuracy, non_extreme_accuracy



    

# Extract features and labels from the loaded dataset for the model ---------------------------------------------------------------------------------------------------------------------
def get_features_labels(ds,all_features):
    """ Given a dataset, extracts features and labels for a extreme, non-extreme classification
    """
    features = ds[all_features].to_array(dim='feature').transpose('time', 'feature').values     #lagged data for each variable. Shape (time steps, number of features) 
    labels = ds['tasmax_extreme_classification'].values                                         #extreme and non-extreme classification fo days 
    valid_indices = ~np.isnan(features).any(axis=1)                                             # remove nan values 
    return features[valid_indices], labels[valid_indices]


# Neural Network class ---------------------------------------------------------------------------------------------------------------------------------------------------------------------




class ExtremeClassifier(nn.Module):                         # ExtremeClassifier is a subclass of nn.Module, which is the base class for all neural networks in PyTorch
    def __init__(self,input_dim):                           # The __init__ method is the constructor of the class
        super(ExtremeClassifier,self).__init__()            # calls the constructor of nn.Module to initialize it properly. Ensures that firstNN inherits all functionalities of nn.Module
        self.fc1 = nn.Linear(input_dim,8)     # First fully connected layer with output size 128 
                                                  #input nº days lagged: tasmax*tasmin*swvl1*swvl2*swvl3*extreme_classification)
        self.fc2 = nn.Linear(8,8)               # Second fully connected layer with input size 128 and output size 8 (Mayer)
        self.fc3 = nn.Linear(8,2)               # Third fully connected layer with input size 8 and output size 2 (Binary classification)

    def forward(self,x):                          # This method defines how the input data flows through the neural network
        if isinstance(x, np.ndarray):  # Check if x is a numpy array
            x = torch.tensor(x, dtype=torch.float32)  # Convert to a tensor

        x = x.view(-1,self.fc1.in_features)                # Flatten the input tensor. Reshapes x from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = F.relu(self.fc1(x))                   # Apply ReLU activation to the output of the first layer 
        x = F.relu(self.fc2(x))                   # Apply ReLU activation to the output of the second layer         
        x = self.fc3(x)                           # Comput the output of the third layer 

        if not self.training: #only probs during evaluation, whee CrossEntropy is not being used and no softmax is called implicit. 
            if not getattr(self, "shap_flag", False): #not apply softam if shap values are being computed 
                x = F.softmax(x, dim=1)        
        return x 


class ExtremeClassifier_modified_threshold(nn.Module):                         # ExtremeClassifier is a subclass of nn.Module, which is the base class for all neural networks in PyTorch
    def __init__(self,input_dim,threshold):                           # The __init__ method is the constructor of the class
        super(ExtremeClassifier_modified_threshold,self).__init__()            # calls the constructor of nn.Module to initialize it properly. Ensures that firstNN inherits all functionalities of nn.Module
        self.fc1 = nn.Linear(input_dim,15)     # First fully connected layer with output size 128 
                                                  #input nº days lagged: tasmax*tasmin*swvl1*swvl2*swvl3*extreme_classification)
        self.fc2 = nn.Linear(15,8)               # Second fully connected layer with input size 128 and output size 8 (Mayer)
        self.fc3 = nn.Linear(8,2)               # Third fully connected layer with input size 8 and output size 2 (Binary classification)

        self.threshold = threshold

    def forward(self,x):                          # This method defines how the input data flows through the neural network
        if isinstance(x, np.ndarray):  # Check if x is a numpy array
            x = torch.tensor(x, dtype=torch.float32)  # Convert to a tensor

        x = x.view(-1,self.fc1.in_features)                # Flatten the input tensor. Reshapes x from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = F.relu(self.fc1(x))                   # Apply ReLU activation to the output of the first layer 
        x = F.relu(self.fc2(x))                   # Apply ReLU activation to the output of the second layer         
        x = self.fc3(x)                           # Comput the output of the third layer 

        if not self.training: #only probs during evaluation, whee CrossEntropy is not being used and no softmax is called implicit. 
            x = F.softmax(x,dim=1)

            #x = (x[:, 1] > self.threshold).float() # will select class 1 (extreme) if the probability is higher than the passed thershold 
        
        return x 
        

class ExtremeClassifier_rawlogit(nn.Module):                         # ExtremeClassifier is a subclass of nn.Module, which is the base class for all neural networks in PyTorch
    def __init__(self,input_dim):                           # The __init__ method is the constructor of the class
        super(ExtremeClassifier_rawlogit,self).__init__()            # calls the constructor of nn.Module to initialize it properly. Ensures that firstNN inherits all functionalities of nn.Module
        self.fc1 = nn.Linear(input_dim,15)     # First fully connected layer with output size 128 
                                                  #input nº days lagged: tasmax*tasmin*swvl1*swvl2*swvl3*extreme_classification)
        self.fc2 = nn.Linear(15,8)               # Second fully connected layer with input size 128 and output size 8 (Mayer)
        self.fc3 = nn.Linear(8,2)               # Third fully connected layer with input size 8 and output size 2 (Binary classification)

    def forward(self,x):                          # This method defines how the input data flows through the neural network
        if isinstance(x, np.ndarray):  # Check if x is a numpy array
            x = torch.tensor(x, dtype=torch.float32)  # Convert to a tensor

        x = x.view(-1,self.fc1.in_features)                # Flatten the input tensor. Reshapes x from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = F.relu(self.fc1(x))                   # Apply ReLU activation to the output of the first layer 
        x = F.relu(self.fc2(x))                   # Apply ReLU activation to the output of the second layer         
        x = self.fc3(x)                           # Comput the output of the third layer 


        return x 




# Simple NN for combining with the CNN 


class ToCombineExtremeClassifier(nn.Module):                         # ExtremeClassifier is a subclass of nn.Module, which is the base class for all neural networks in PyTorch
    def __init__(self,input_dim, train_alone_NN = True, num_classes = 2):                           # The __init__ method is the constructor of the class
        super(ToCombineExtremeClassifier,self).__init__()            # calls the constructor of nn.Module to initialize it properly. Ensures that firstNN inherits all functionalities of nn.Module

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.BatchNorm1d(15),
            nn.ReLU(),
            nn.Linear(15, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            #nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            #nn.ReLU()
        )

        self.train_alone_NN = train_alone_NN

        if self.train_alone_NN:
            self.final_classification = nn.Linear(8, num_classes) # From hidden dim to output classes

    def forward(self, x):                         
        if isinstance(x, np.ndarray): 
            x = torch.tensor(x, dtype=torch.float32)  

        x = x.view(-1, self.fc_layers[0].in_features) 
        hidden = self.fc_layers(x) 

        if self.train_alone_NN:
            return  self.final_classification(hidden) #return classification only with NN if training NN alone 
        else:                 
            return hidden



# CNN class -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class original_CNN_Era5ExtemeClassifer(nn.Module):
    def __init__(self, lat_size,lon_size, num_lags, num_classes=2):
        """
        CNN model to process ERA5 lagged data (lat,lon)
        
        Args:
            lat_size : number of latitude points present in the dataset
            lon_size : number of longitude points present in the dataset 
            num_lags (int): Number of lagged time steps (acts as channels in Conv layers).
            num_classes (int): Number output classes (2 since it's binary classification).
        """
        super(original_CNN_Era5ExtemeClassifer, self).__init__() #initiate
        
  
        #each laged variable is an input 
     
        self.conv1 = nn.Conv2d(in_channels=num_lags, out_channels=32, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #spatial size transformed to   (num lat points / 8, num lon points / 8)
        
       
        self.fc1 = nn.Linear(128 * (lat_size // 8) * (lon_size // 8), 256)  #
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # binary classification
        
    def forward(self, x):
        """ Forward pass through CNN model.
        x : batch_size, num_lags, lat, lon
        """
        x = self.pool(F.relu(self.conv1(x))) # ReLU as activation function. Apply convolution layer, ReLU activation , and pool to make effective the convolution
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        
        # Flatten before FC layers
        x = x.view(x.size(0), -1)     
        
        hidden = F.relu(self.fc1(x))  # fully connected layer - 1
        out = self.fc2(hidden) 
        
        return hidden, out 





# Batch norm: before the activation function, normalized the inputs of a layer acorss the batch. Reduced changes in layer inputs during training. Inputs now have mean = 0 and std = 1 


class CNN_Era5ExtemeClassifer(nn.Module):
    def __init__(self, lat_size, lon_size, num_variables, num_lags, train_alone_CNN=False, num_classes=2,pretrain_flag=False):
        """ 
        num_lags: nuber entering features 
        num_classes: output"""
        super(CNN_Era5ExtemeClassifer, self).__init__()

        self.lat_size = lat_size
        self.lon_size = lon_size
        self.num_variables = num_variables
        self.num_lags = num_lags
        self.pretrain_flag = pretrain_flag

        self.train_alone_CNN = train_alone_CNN

        # Feature extractors 

        #if pretrain_flag:
        #    self.conv_block1 = nn.Sequential(
        #     #apply 32 filters of shape (num_lags,lat,lon)
        #    
        #        nn.Conv2d(in_channels=num_variables, out_channels=32, kernel_size=2, stride=1, padding='same',dilation=1), # creates 32 maps. padding=1 keeps the spatial size the same.             
        #    nn.BatchNorm2d(32), #normalizes each channel
        #    nn.ReLU(), #non-linear activation 
        #        # No pooling
        #    )
#
        #else:
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_variables, out_channels=8, kernel_size=2, stride=1, padding='same',dilation=1), # creates 32 maps. padding=1 keeps the spatial size the same. 
            nn.BatchNorm2d(8), #normalizes each channel
            nn.ReLU(), #non-linear activation 
            # No pooling
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same', dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pool after model has seen enouch local structures
                                                   # increase receptive field 
                                                   # add translation invariance 
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same',dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same', dilation=1), 
            #dilation=1 : normal convolution
            #dilation=2 : skipping one pizel per input 
            #dilation=4 : skiping 3 pixels
            nn.BatchNorm2d(64),
            nn.ReLU(),
          # nn.MaxPool2d(kernel_size=2, stride=2)  # modest final downsampling
        )

        #Spatial attention layers. Learn from regions and save maps of explainability 
        self.attn1 = functions_improve_CombinedModel.SpatialAttention(32)
        self.attn2 = functions_improve_CombinedModel.SpatialAttention(64)
        self.attn3 = functions_improve_CombinedModel.SpatialAttention(128)

        
        with torch.no_grad():
            # add 2 to number self.number_variables if info of lat and lon is given 
            dummy_input_single_lag = torch.zeros(1, self.num_variables, self.lat_size, self.lon_size)
            dummy_output = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(dummy_input_single_lag))))
            #dummy_output_attn = self.attn3(dummy_output)
            flattened_size_per_lag = dummy_output.view(1, -1).shape[1]


        self.fc1 = nn.Linear(self.num_lags * flattened_size_per_lag, 64)  

        if self.train_alone_CNN:
            self.final_classification = nn.Linear(64, num_classes) # From hidden dim to output classes

    def forward(self, x ):

        batch_size = x.size(0) #batch size. shape x: (batch_size, num_features, lat_size, lon_size)
          
        
        lag_outputs_flattened = [] # to store and flatten lags

        #if not self.pretrain_flag:
       
        for t in range(self.num_lags):

            lag_data = x[:, :, t, :, :]

            # (batch_size, num_variables + 2, lat_size, lon_size)
            # in case of passing lat and lon info 
            #lag_data_with_pos = torch.cat([lag_data, lat, lon], dim=1)

            conv_out = self.conv_block1(lag_data)
            conv_out = self.conv_block2(conv_out)
            conv_out = self.conv_block3(conv_out)
            conv_out = self.conv_block4(conv_out)
            
            #attn_out = self.attn3(conv_out)

            # (batch_size, flattened_size_per_lag)
            flattened_out = conv_out.view(batch_size, -1)
            lag_outputs_flattened.append(flattened_out)

        # concatenation of the different lags 
        # (batch_size, num_lags * flattened_size_per_lag)
        combined_output = torch.cat(lag_outputs_flattened, dim=1)

      
        hidden = F.relu(self.fc1(combined_output))

        if self.train_alone_CNN:
            return self.final_classification(hidden)

        else:
            return hidden 



# Merged model CNN + simple NN class ----------------------------------------------------------------------------------------------------------------------------------------------------

# NN output + CNN output -> concatenation -> FC1 -> ReLU -> FC2 → ReLU -> FC3 -> ReLU -> output

class CombinedModel(nn.Module):
    def __init__(self, nn_model, cnn_model, nn_hidden_dim, cnn_hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.nn_model = nn_model
        self.cnn_model = cnn_model

        #self.nn_proj = nn.Linear(nn_hidden_dim, 64)    # optional
        #self.cnn_proj = nn.Linear(cnn_hidden_dim, 64)  # optional
        
        #combined_hidden_size = 128  # 64 + 64          # optional

        combined_hidden_size =  cnn_hidden_dim + nn_hidden_dim 

        ### Optional interaction: need to define interaction_hidden_dim in __init__ if wanted to use 
        #self.interaction = functions_improve_CombinedModel.InteractionLayer(
        #    nn_dim=nn_hidden_dim, 
        #    cnn_dim=cnn_hidden_dim,
        #    hidden_dim = interaction_hidden_dim
        #)

        self.fc_layers = nn.Sequential(
            nn.Linear(combined_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(8,output_dim)
        )
        
    def forward(self, nn_input, cnn_input):
        nn_hidden = self.nn_model(nn_input)
        cnn_hidden = self.cnn_model(cnn_input)

        #nn_hidden, cnn_hidden = self.interaction(nn_hidden, cnn_hidden)
        combined = torch.cat((nn_hidden, cnn_hidden), dim=1)

        output = self.fc_layers(combined)

        return output


#Custom dataset ERA5 data ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self, local_data, large_data):
        
        self.local_data = local_data
        self.large_data = large_data

        self._test_coherence()
        
        
    def _test_coherence(self):
        """
        Check if the local and large data have the same time values
        """
        for t in ["day", "month", "year"]: 
            for v in ["ds_g200", "ds_g500", "ds_psl"]:
                _times1 = getattr(self.local_data.ds.time.dt, t).values
                _times2 = getattr(self.large_data, v).time.dt
                _times2 = getattr(_times2, t).values
                
                assert (_times1 == _times2).all(), f"Time values do not match for {t} in {v}"
        
    def __len__(self):
        return len(self.local_data.labels)
    
    def __getitem__(self, idx):
        
        local_features = self.local_data[idx][0]
        large_features = self.large_data[idx]
        local_labels = self.local_data[idx][1]
        
        return local_features, large_features, local_labels 

class CombinedDataset_single_field(torch.utils.data.Dataset):

    def __init__(self, local_data, large_data):
        
        self.local_data = local_data
        self.large_data = large_data

        self._test_coherence()
        
        
    def _test_coherence(self):
        """
        Check if the local and large data have the same time values
        """
        for t in ["day", "month", "year"]: 
            for v in ["ds"]:
                _times1 = getattr(self.local_data.ds.time.dt, t).values
                _times2 = getattr(self.large_data, v).time.dt
                _times2 = getattr(_times2, t).values
                
                assert (_times1 == _times2).all(), f"Time values do not match for {t} in {v}"
        
    def __len__(self):
        return len(self.local_data.labels)
    
    def __getitem__(self, idx):
        
        local_features = self.local_data[idx][0]
        large_features = self.large_data[idx]
        local_labels = self.local_data[idx][1]
        
        return local_features, large_features, local_labels 

# -------------------------------------------------------------

class ERA5Dataset_extremes(Dataset):
    def __init__(self, file_g500,file_g200,file_psl, start_date, end_date, months, start_lag, lags_era5,transform=None):
        """
        Args:
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            start_lag (int) : 1 for the lag 1. 
            lags_era5 (int) : number of lags to take in the dataset
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
            transform (callable, optional): Optional transform to be applied.
        """

        # Select time period and months 
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
        self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
        self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        self.ds_g200 = self.ds_g200.squeeze('plev', drop=True) #remove plev dimension
        
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
        self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)

       #self.ds_hus700 = xr.open_dataset(file_hus700).sel(time=slice(start_date, end_date))
       #self.ds_hus700 = self.ds_hus700.sel(time=self.ds_hus700.time.dt.month.isin(months), drop=True)
#
       #self.ds_hus850 = xr.open_dataset(file_hus850).sel(time=slice(start_date, end_date))
       #self.ds_hus850 = self.ds_hus850.sel(time=self.ds_hus850.time.dt.month.isin(months), drop=True)

       #self.ds_hus975 = xr.open_dataset(file_hus975).sel(time=slice(start_date, end_date))
       #self.ds_hus975 = self.ds_hus975.sel(time=self.ds_hus975.time.dt.month.isin(months), drop=True)
#
       #self.ds_rsds = xr.open_dataset(file_rsds).sel(time=slice(start_date, end_date))
       #self.ds_rsds = self.ds_rsds.sel(time=self.ds_rsds.time.dt.month.isin(months), drop=True)
        
        # Identify lagged variables if not provided

        self.lagged_vars = {
            'g500': [f'lagged_era5g500_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1) ],
            'g200': [f'lagged_era5g200_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1) ],
            'psl': [f'lagged_era5psl_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1) ],
            #'hus700': [f'lagged_era5hus700_anomalies_lag{lag}' for lag in range(1, lags_era5)],
            #'hus850': [f'lagged_era5hus850_anomalies_lag{lag}' for lag in range(1, lags_era5)],
            #'hus975': [f'lagged_era5hus975_anomalies_lag{lag}' for lag in range(1, lags_era5)],
            #'rsds': [f'lagged_era5rsds_anomalies_lag{lag}' for lag in range(1, lags_era5)]
        }
       
        self.lagged_vars_g500 = self.lagged_vars['g500']
        self.lagged_vars_g200 = self.lagged_vars['g200']
        self.lagged_vars_psl = self.lagged_vars['psl']
        #self.lagged_vars_hus700 = self.lagged_vars['hus700']
        #self.lagged_vars_hus850 = self.lagged_vars['hus850']
        #self.lagged_vars_hus975 = self.lagged_vars['hus975']
        #self.lagged_vars_rsds = self.lagged_vars['rsds']
        lagged_vars = self.lagged_vars_g500 + self.lagged_vars_g200 + self.lagged_vars_psl #+ self.lagged_vars_hus850 + self.lagged_vars_hus700#+ self.lagged_vars_hus975 + self.lagged_vars_rsds
            
        self.all_features = lagged_vars
        
        
        self.features_g500 = self.ds_g500[self.lagged_vars_g500].to_array().transpose('time', 'variable','lat', 'lon')
        self.features_g200 = self.ds_g200[self.lagged_vars_g200].to_array().transpose('time', 'variable','lat', 'lon')
        self.features_psl = self.ds_psl[self.lagged_vars_psl].to_array().transpose('time', 'variable','lat', 'lon')
        #self.features_hus700 = self.ds_hus700[self.lagged_vars_hus700].to_array().transpose('time', 'variable','lat', 'lon')
        #self.features_hus850 = self.ds_hus850[self.lagged_vars_hus850].to_array().transpose('time', 'variable','lat', 'lon')
        #self.features_hus975 = self.ds_hus975[self.lagged_vars_hus975].to_array().transpose('time', 'variable','lat', 'lon')
        #self.features_rsds = self.ds_rsds[self.lagged_vars_rsds].to_array().transpose('time', 'variable','lat', 'lon')

        

        self.features = np.concatenate(
            [self.features_g500.values, self.features_g200.values, self.features_psl.values],axis=1)
            #[self.features_g500.values, self.features_g200.values, self.features_psl.values, self.features_hus850.values, self.features_hus700.values],axis=1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (num_lags, lat, lon)
        
        return sample


# Only g500 class ---------------------------------------------------------------------------------------------------------------------------------------------------

class ERA5Dataset_extremes_single_field(Dataset):
    def __init__(self, file, variable,start_date, end_date, months, lags_era5,transform=None):
        """
        Args:
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
            transform (callable, optional): Optional transform to be applied.
        """

        self.variable = variable
        
        # Select time period and months 
        self.ds = xr.open_dataset(file).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months), drop=True)
        if variable == 'g200':
            self.ds = self.ds.squeeze('plev', drop=True) #remove plev dimension
        
        # Identify lagged variables if not provided

        self.lagged_vars_dict = {
            variable: [f'lagged_era5{variable}_anomalies_lag{lag}' for lag in range(1, lags_era5) ]
        }
       
        self.lagged_vars = self.lagged_vars_dict[self.variable]
      
        lagged_vars = self.lagged_vars
            
        self.all_features = lagged_vars
        
        
        self.features = self.ds[self.lagged_vars].to_array().transpose('time', 'variable','lat', 'lon')
  
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.features[idx].values.astype('float32'))
        
        return sample


# Sequential Custom Class ERA5 ---------------------------------------------------------------------------------------------------------------------------------------


class ERA5Dataset_extremes_Sequential(Dataset):
    def __init__(self, file_g500,file_g200,file_psl, start_date, end_date, months, lags_era5,transform=None):
        """
        Args:
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
            transform (callable, optional): Optional transform to be applied.
        """

        # Select time period and months 
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date))
        self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date))
        self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        self.ds_g200 = self.ds_g200.squeeze('plev', drop=True) #remove plev dimension
        
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date))
        self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)


        # Identify lagged variables if not provided

        self.lagged_vars = {
            'g500': [f'lagged_era5g500_anomalies_lag{lag}' for lag in range(1, lags_era5) ],
            'g200': [f'lagged_era5g200_anomalies_lag{lag}' for lag in range(1, lags_era5) ],
            'psl': [f'lagged_era5psl_anomalies_lag{lag}' for lag in range(1, lags_era5) ],
        }
       
        self.lagged_vars_g500 = self.lagged_vars['g500']
        self.lagged_vars_g200 = self.lagged_vars['g200']
        self.lagged_vars_psl = self.lagged_vars['psl']

        lagged_vars = self.lagged_vars_g500 + self.lagged_vars_g200 + self.lagged_vars_psl 
            
        self.all_features = lagged_vars
        
        
        self.features_g500 = self.ds_g500[self.lagged_vars_g500].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')
        self.features_g200 = self.ds_g200[self.lagged_vars_g200].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')
        self.features_psl = self.ds_psl[self.lagged_vars_psl].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')

        

        self.features = np.stack(
            [self.features_g500.values, self.features_g200.values, self.features_psl.values],axis=1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (num_lags, lat, lon)
        
        return sample






# --------------------------------------------------------------------------------------------------------------------------------------------------------------------



        


# Custom Dataset for ERA5land ------------------------------------------------------------------------------------------------------------------------------------------------------------

import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ERA5LandDataset_extremes_location(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, start_date, end_date, months,scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) #open selected months 

        # Define lagged variables
        self.lagged_vars = {
            #'tasmax_anomalies':[f'lagged_era5_land_tasmax_anomalies_lag{i}' for i in range(1,4)],
            #'tasmin_anomalies':[f'lagged_era5_land_tasmin_anomalies_lag{i}' for i in range(1,4)],
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)]
        }

        # Collect feature names
        self.all_features = [f for vars in self.lagged_vars.values() for f in vars]

        # Extract features (lagged-data in each time-step) and labels(exteme/non-extreme)
        self.features = self.ds[self.all_features].to_array(dim='feature').transpose('time', 'feature').values
        self.labels = self.ds['tasmax_extreme_classification'].values

        # Remove NaN values from samples 
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]

        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 

## Dataset only for count observational data-------------------------------------------------------------------------------------------------------

class Dataset_count_observational_TX(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, start_date, end_date, months,scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) #open selected months 
        
        #self.features = self.ds[self.all_features].to_array(dim='feature').transpose('time', 'feature').values
        self.labels = self.ds['tx_mean_extreme_classification'].values

        # Remove NaN values from samples 

        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 

    

## -----------------------------------------------------------------------------------------------------------------------------------------------


class ERA5LandDataset_extremes_location_swvl_averaged(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, start_date, end_date, months,scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) #open selected months 

        # Define lagged variables
        self.lagged_vars = {
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Collect feature names
        self.all_features = [var for var in self.lagged_vars.keys()] 

        # Extract features (lagged-data in each time-step) and labels(exteme/non-extreme)

        swvl_avg_features = []
        
        for key in self.lagged_vars:
            swvl_lags = self.ds[self.lagged_vars[key]].to_array(dim='lag').transpose('time', 'lag').values
            swvl_avg = np.nanmean(swvl_lags, axis=1)  
            swvl_avg_features.append(swvl_avg)       
        
        features_loc = np.stack(swvl_avg_features, axis=1)
        self.features = features_loc
    
        
        #self.features = self.ds[self.all_features].to_array(dim='feature').transpose('time', 'feature').values
        self.labels = self.ds['tasmax_extreme_classification'].values

        # Remove NaN values from samples 
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]

        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 

    

class ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, file_CO2, start_date, end_date, months,scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) #open selected months 
        self.dsco2 = xr.open_dataset(file_CO2).sel(time=slice(start_date, end_date))
        self.co2conc = self.dsco2['co2_concentration'].values

        # Define lagged variables
        self.lagged_vars = {
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Collect feature names
        self.all_features = ['co2'] + [var for var in self.lagged_vars.keys()] 

        # Extract features (lagged-data in each time-step) and labels(exteme/non-extreme)

        swvl_avg_features = []
        
        for key in self.lagged_vars:
            swvl_lags = self.ds[self.lagged_vars[key]].to_array(dim='lag').transpose('time', 'lag').values
            swvl_avg = np.nanmean(swvl_lags, axis=1)  
            swvl_avg_features.append(swvl_avg)       
        
        self.features_loc = np.stack(swvl_avg_features, axis=1)
        
        self.co2conc = self.co2conc.reshape(-1,1)
        self.features = np.concatenate([self.co2conc,self.features_loc],axis=1)
    
        
        #self.features = self.ds[self.all_features].to_array(dim='feature').transpose('time', 'feature').values
        self.labels = self.ds['tasmax_extreme_classification'].values

        # Remove NaN values from samples 
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]

        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 

    


class ERA5LandDataset_extremes_eofs(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path_loc, file_g500, file_g200, file_psl, start_date, end_date, months, lags_eof,scaler=None):
        """
        Args:
            file_path_loc (str): Path to the NetCDF file containing ERA5 land dataset.
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        
        # Load ERA5-Land data
        self.ds_loc = xr.open_dataset(file_path_loc).sel(time=slice(start_date, end_date)).isel(time=slice(3,None))
        
        # Load EOF datasets
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date))
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date))
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date))

        # Apply month filtering
        self.ds_loc = self.ds_loc.sel(time=self.ds_loc.time.dt.month.isin(months), drop=True)
        #self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        #self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        #self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)

        num_efos_variable = {
            'tasmax': 30,
            'tasmin': 30,
            'g200': 23,
            'g500': 28,
            'psl': 36
        }
        

        # Define lagged variables for ERA5-Land
        self.lagged_vars = {
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Define EOF features (each lag variable contains EOFs)
        self.eof_lagged_vars = {
            'g500': [f'pcs_g500_lag{lag}' for lag in range(1, lags_eof) ],
            'g200': [f'pcs_g200_lag{lag}' for lag in range(1, lags_eof) ],
            'psl': [f'pcs_psl_lag{lag}' for lag in range(1, lags_eof) ],
        }

        # Collect all feature names (ignoring EOFs for now, as they need special extraction)
        features_loc = [f for vars in self.lagged_vars.values() for f in vars]
        self.features_location = features_loc
        features_region = [f"{f}_eof{i}" for vars in self.eof_lagged_vars.values() for f in vars for i in range(1, num_efos_variable[f.split("_")[1]]+1)]
        self.features_region = features_region
        self.all_features = features_loc + features_region  

        # ERA5-Land features 
        features_loc = self.ds_loc[self.features_location].to_array(dim='feature').transpose('time', 'feature').values

        # Extract EOFs correctly: select first dimension (EOF index) separately
        def extract_eofs(ds, variable_list,num_eofs):
            """Extracts the first 5 EOFs from each lag variable."""
            eof_features = []
            for var in variable_list: #select the variable 
                #for i in range(num_eofs): #select the eof
                eof_features.append(ds[var].values[:num_eofs, :].T)  # Transpose to match (time, feature)
            return np.concatenate(eof_features, axis=1)  # Merge along feature axis

        # Extract EOF features from each dataset. each eof of each lag will be a feature, a vector with time dimensions 
        features_g500 = extract_eofs(self.ds_g500, self.eof_lagged_vars['g500'],num_efos_variable['g500'])
        features_g200 = extract_eofs(self.ds_g200, self.eof_lagged_vars['g200'],num_efos_variable['g200'])
        features_psl = extract_eofs(self.ds_psl, self.eof_lagged_vars['psl'],num_efos_variable['psl'])

        # Concatenate all features along the feature dimension
        self.features = np.concatenate([features_loc, features_g500, features_g200, features_psl], axis=1)

        # Extract labels
        self.labels = self.ds_loc['tasmax_extreme_classification'].values

        # Remove NaN values (valid indices only)
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]


        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 


# Class averaging the lags in the soil moisture -----------------------------------------------------------------------------------------

class ERA5LandDataset_extremes_eofs_swvl_average(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path_loc, file_g500, file_g200, file_psl, start_date, end_date, months, lags_eof, scaler=None):
        """
        Args:
            file_path_loc (str): Path to the NetCDF file containing ERA5 land dataset.
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        
        # Load ERA5-Land data
        self.ds_loc = xr.open_dataset(file_path_loc).sel(time=slice(start_date, end_date)).isel(time=slice(3,None))
        
        # Load EOF datasets
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date))
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date))
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date))

        # Apply month filtering
        self.ds_loc = self.ds_loc.sel(time=self.ds_loc.time.dt.month.isin(months), drop=True)
        #self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        #self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        #self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)

        num_efos_variable = {
            'tasmax': 30,
            'tasmin': 30,
            'g200': 23,
            'g500': 28,
            'psl': 36
        }
        

        # Define lagged variables for ERA5-Land
        self.lagged_vars = {
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Define EOF features (each lag variable contains EOFs)
        self.eof_lagged_vars = {
            'g500': [f'pcs_g500_lag{lag}' for lag in range(1, lags_eof) ],
            'g200': [f'pcs_g200_lag{lag}' for lag in range(1, lags_eof) ],
            'psl': [f'pcs_psl_lag{lag}' for lag in range(1, lags_eof) ],
        }

        # Collect all feature names (ignoring EOFs for now, as they need special extraction)
        features_loc = [var for var in self.lagged_vars.keys()]  #selects only the key in the dictionary (swvl1_anomalies,swvl2_anomalies,swvl3_anomalies)
        self.features_location = features_loc
        features_region = [f"{f}_eof{i}" for vars in self.eof_lagged_vars.values() for f in vars for i in range(1, num_efos_variable[f.split("_")[1]]+1)]
        self.features_region = features_region
        self.all_features = features_loc + features_region  

        # ERA5-Land features 
        swvl_avg_features = []

        for key in self.lagged_vars:
            swvl_lags = self.ds_loc[self.lagged_vars[key]].to_array(dim='lag').transpose('time', 'lag').values
            swvl_avg = np.nanmean(swvl_lags, axis=1)  
            swvl_avg_features.append(swvl_avg)       
        
        features_loc = np.stack(swvl_avg_features, axis=1)

        # Extract EOFs correctly: select first dimension (EOF index) separately
        def extract_eofs(ds, variable_list,num_eofs):
            """Extracts the first 5 EOFs from each lag variable."""
            eof_features = []
            for var in variable_list: #select the variable 
                #for i in range(num_eofs): #select the eof
                eof_features.append(ds[var].values[:num_eofs, :].T)  # Transpose to match (time, feature)
            return np.concatenate(eof_features, axis=1)  # Merge along feature axis

        # Extract EOF features from each dataset. each eof of each lag will be a feature, a vector with time dimensions 
        features_g500 = extract_eofs(self.ds_g500, self.eof_lagged_vars['g500'],num_efos_variable['g500'])
        features_g200 = extract_eofs(self.ds_g200, self.eof_lagged_vars['g200'],num_efos_variable['g200'])
        features_psl = extract_eofs(self.ds_psl, self.eof_lagged_vars['psl'],num_efos_variable['psl'])

        # Concatenate all features along the feature dimension
        self.features = np.concatenate([features_loc, features_g500, features_g200, features_psl], axis=1)

        # Extract labels
        self.labels = self.ds_loc['tasmax_extreme_classification'].values

        # Remove NaN values (valid indices only)
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]


        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 


class ERA5LandDataset_extremes_eofs_swvl_average_CO2(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path_loc, file_CO2, file_g500, file_g200, file_psl, start_date, end_date, months, lags_eof, scaler=None):
        """
        Args:
            file_path_loc (str): Path to the NetCDF file containing ERA5 land dataset.
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        
        # Load ERA5-Land data
        self.ds_loc = xr.open_dataset(file_path_loc).sel(time=slice(start_date, end_date)).isel(time=slice(3,None))

        self.dsco2 = xr.open_dataset(file_CO2).sel(time=slice(start_date, end_date))
        self.co2conc = self.dsco2['co2_concentration'].values
        self.co2conc = self.co2conc.reshape(-1,1)
        
        # Load EOF datasets
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date))
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date))
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date))

        # Apply month filtering
        self.ds_loc = self.ds_loc.sel(time=self.ds_loc.time.dt.month.isin(months), drop=True)
        self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)

        num_efos_variable = {
            'tasmax': 30,
            'tasmin': 30,
            'g200': 23,
            'g500': 28,
            'psl': 36
        }
        

        # Define lagged variables for ERA5-Land
        self.lagged_vars = {
            'swvl1_anomalies': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2_anomalies': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3_anomalies': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Define EOF features (each lag variable contains EOFs)
        self.eof_lagged_vars = {
            'g500': [f'pcs_g500_lag{lag}' for lag in range(1, lags_eof) ],
            'g200': [f'pcs_g200_lag{lag}' for lag in range(1, lags_eof) ],
            'psl': [f'pcs_psl_lag{lag}' for lag in range(1, lags_eof) ],
        }

        # Collect all feature names (ignoring EOFs for now, as they need special extraction)
        features_loc = [var for var in self.lagged_vars.keys()]  #selects only the key in the dictionary (swvl1_anomalies,swvl2_anomalies,swvl3_anomalies)
        self.features_location = features_loc
        features_region = [f"{f}_eof{i}" for vars in self.eof_lagged_vars.values() for f in vars for i in range(1, num_efos_variable[f.split("_")[1]]+1)]
        self.features_region = features_region
        self.all_features = ['co2'] + features_loc + features_region  

        # ERA5-Land features 
        swvl_avg_features = []

        for key in self.lagged_vars:
            swvl_lags = self.ds_loc[self.lagged_vars[key]].to_array(dim='lag').transpose('time', 'lag').values
            swvl_avg = np.nanmean(swvl_lags, axis=1)  
            swvl_avg_features.append(swvl_avg)       
        
        features_loc = np.stack(swvl_avg_features, axis=1)

        # Extract EOFs correctly: select first dimension (EOF index) separately
        def extract_eofs(ds, variable_list,num_eofs):
            """Extracts the first 5 EOFs from each lag variable."""
            eof_features = []
            for var in variable_list: #select the variable 
                #for i in range(num_eofs): #select the eof
                eof_features.append(ds[var].values[:num_eofs, :].T)  # Transpose to match (time, feature)
            return np.concatenate(eof_features, axis=1)  # Merge along feature axis

        # Extract EOF features from each dataset. each eof of each lag will be a feature, a vector with time dimensions 
        features_g500 = extract_eofs(self.ds_g500, self.eof_lagged_vars['g500'],num_efos_variable['g500'])
        features_g200 = extract_eofs(self.ds_g200, self.eof_lagged_vars['g200'],num_efos_variable['g200'])
        features_psl = extract_eofs(self.ds_psl, self.eof_lagged_vars['psl'],num_efos_variable['psl'])

        # Concatenate all features along the feature dimension
        self.features = np.concatenate([self.co2conc,features_loc, features_g500, features_g200, features_psl], axis=1)

        # Extract labels
        self.labels = self.ds_loc['tasmax_extreme_classification'].values

        # Remove NaN values (valid indices only)
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]


        # Standardization with StandardScaler if decided to be used 
        #if scaler is None:
        #    self.scaler = StandardScaler()
        #    self.features = self.scaler.fit_transform(self.features)  # Fit & transform on train data
        #else:
        #    self.scaler = scaler
        #    self.features = self.scaler.transform(self.features)  # Use existing scaler for test data

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      # Each idx corresponds to a time step. So each sample is a 27 features values + 1 label value
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() # squeeze labels to get a 1D array 


#---------------------------------------------------------------------------------------------------------------------------------


# Train Models --------------------------------------------------------------------------------

# Train EOFsNN 

def train_simple_NN_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,plot_loss, print_loss, early_stop, patience = 5,print_early_stop=True):

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  

    losses_train = []
    losses_val = []
    #out_probabilitites = []

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # training 

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    num_e = 0
    with tqdm(range(num_epochs), position=0, leave=True) as pbar:
        for _ in pbar:
            total_loss = 0
            train_batches = 0
    
            # For validation
            correct_val_0 = 0    # Correct predictions for class 0
            correct_val_1 = 0    # Correct predictions for class 1
            total_val_0 = 0      # Total samples for class 0
            total_val_1 = 0      # Total samples for class 1
        
    
            for inputs,labels in train_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                # Only use local and labels 
                #print(f"inputs: {inputs.device}, labels: {labels.device}, model params: {next(model.parameters()).device}, criterion weight: {criterion.weight.device}")
                optimizer.zero_grad()
                outputs = model(inputs)
    
                #out_probabilitites.append(outputs)
                
                loss = criterion(outputs, labels)
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
                for inputs,labels in val_loader:
                    val_inputs,val_labels = inputs.to(device), labels.to(device)
                    
                    val_outputs = model(val_inputs)  
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

    return losses_train, losses_val #, F.softmax(torch.cat(out_probabilitites),dim=1)



    
# For training the NN ****************************************************************************

def train_NNmodel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, print_loss, early_stop, patience = 5,print_early_stop=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  

    losses_train = []
    losses_val = []
    #out_probabilitites = []

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

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
    
                #out_probabilitites.append(outputs)
                
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


# For training the CNN *******************************************************************



def train_CNNmodel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, print_loss, early_stop, patience = 5,print_early_stop=True):

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



# For training the combined model ****************************************************************************

def train_CombinedModel(model, combined_train_loader, combined_val_loader, criterion, optimizer, num_epochs,plot_loss, 
                        print_loss, early_stop, patience = 5,print_early_stop=True, trial=None):

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


