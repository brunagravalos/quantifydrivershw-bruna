# ======================================================================================================
# IMPORT NEEDED PACKAGES
# ======================================================================================================

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
import pandas as pd
import torch
import xarray as xr
import numpy as np
from skimage import io, transform
import random
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn                   
import torch.nn.functional as F        
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re 
import shap 
from sklearn.metrics import balanced_accuracy_score
import pickle
import optuna
from sklearn.metrics import f1_score
import gc
import tqdm
import argparse

import functions_ML
from functions_ML import ERA5LandDataset_extremes_location_spei
import convnext_functions

# ======================================================================================================

# DEFINE DEVICE ----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------

# CECK DETERMINISM 

try:
    torch.use_deterministic_algorithms(True)
    print("Using deterministic algorithms.")
except Exception as e:
    print(f"Could not enforce deterministic algorithms: {e}")

def check_seeds():
    print(f"Torch seed: {torch.initial_seed()}")
    print(f"NumPy seed: {np.random.get_state()[1][0]}")
    print(f"Python random seed: {random.getstate()[1][0]}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
check_seeds()


# ==================================================================================================================================
# 1. Global Configuration
#===================================================================================================================================
#File paths ERA5 data -----------------------------------------------------------------------------------------------------------------------------------------------------
        
file_g500 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_g500_1x1_lagged_standarized_anomalies.nc"
file_g200 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_g200_1x1_lagged_standarized_anomalies.nc"
file_psl = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_psl_1x1_lagged_standarized_anomalies.nc"

# File local scale data and extreme classification ------------------------------------------------
file_local_scale = f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{percentile_to_load}_{site}_lagged_standarized_anomalies_and_extreme_detection.nc"
    
# File CO2 data 
file_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"


# Percentile that is used to define the extremes in the local scale data
percentile_to_load = '90p' 
# SPEI/SPI configuration 
spei_spi = 'spei'  # 'spi' or 'spei'
scales_spei = ['30','60']
distribution = 'gamma'

# =================================================================================================================
# Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
# =================================================================================================================

HYPMS = dict(
    epochs= 75,
    lr= 1e-4,
    w_decay= 0.01,
)

# Start date for all datasets
start_date = "1950-01-01"
# Variables large-scale and local scale to use 
variables_era5 = ['g500','g200','psl']
    
# Local-scale datasets configuration
_ERA5LAND_TRAIN_DATASET_CONF = dict(
start_date= start_date,
end_date= "2013-12-31",
months = [6,7,8]
    )

_ERA5LAND_TEST_DATASET_CONF = dict(
start_date= "2014-01-01",
end_date= "2023-12-31",
months = [6,7,8]         
    )

# Large-scale datasets configuration
_ERA5_TRAIN_DATASET_CONF = dict(
start_date= start_date,
end_date= "2013-12-31",
start_lag = 1,
lags_era5 = 1,
months = [6,7,8],
variables = variables_era5
    )

_ERA5_TEST_DATASET_CONF = dict(
start_date= "2014-01-01",
end_date= "2023-12-31",
start_lag = 1,
lags_era5 = 1,
months = [6,7,8],
variables = variables_era5   
    )

# Number of lags for the larg-scale data ----------------------------------------------------------------------------------------------------------------------------------
number_lags = _ERA5_TEST_DATASET_CONF['lags_era5']


#===========================================================================================================================================================================
# DETERMINSIM, SEED CREATION AND SEED RESTART 
#===========================================================================================================================================================================

# Function to verify determinism -----------------------------------------------------------------------------------------------------------------------------------------

def verify_determinism():
    # Check PyTorch
    print(f"PyTorch rand(): {torch.rand(1).item()}")  # Should match across runs
    
    # Check NumPy
    print(f"NumPy rand(): {np.random.rand()}")  # Should match
    
    # Check Python random
    print(f"Python random(): {random.random()}")  # Should match

# Function to Reset the seed, for determinsim in the computations *********************************************************************************************************

def reset_seeds(seed=42):
    g = torch.Generator()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)  # For DataLoader's generator


    # Generate list of seeds for the ensamble loop ***************************************************************************************************************************

def generate_ensemble_seeds(fixed_seed=123):
    rng = np.random.default_rng(fixed_seed) #generator
    seeds = rng.integers(low=0, high=2**32 - 1, size=1).tolist()
    return seeds


# Generate list of seeds for the ensamble ---------------------------------------------------------------

list_seeds = generate_ensemble_seeds(fixed_seed=123)


#==============================================================================================================================================================================
#OPTUNA OBJECTIVE FUNCTION 
#==============================================================================================================================================================================

def objective(trial, site, seed, train_subset_combined, val_subset_combined):
    # 1. Hyperparameters for this trial
    # =================================================================================
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    w_decay = trial.suggest_float("w_decay", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    minority_weight_multiplier = trial.suggest_float("minority_weight_multiplier", 0.1, 8.0)

    # =================================================================================================
    # Dataloaders setup
    # =================================================================================

    reset_seeds(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    _DATALOADERS_CONF = dict(
        batch_size=batch_size, 
        drop_last=False,
        shuffle=True,
        num_workers=4,
        generator=g,
        pin_memory=True,  # Use pinned memory for faster data transfer to GPU
    )
    _DATALOADERS_VAL_CONF = dict(
        batch_size=batch_size, 
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,  # Use pinned memory for faster data transfer to GPU
    )   

    reset_seeds(seed)

    combined_train_loader = DataLoader(train_subset_combined, **_DATALOADERS_CONF)
    combined_val_loader = DataLoader(val_subset_combined, **_DATALOADERS_VAL_CONF)

    # 3. Setup Model, Optimizer and Loss with suggested hyperparameters
    # =================================================================================
    # Class weights
    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    base_minority_weight = class_counts[0] / class_counts[1]
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=smoothed_weights, reduction='mean').to(device)

    # Initialize models
    reset_seeds(seed)
    NN_model = functions_ML.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    CNN_model = convnext_functions.ConvNext(
        num_channels=len(train_features_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False
    ).to(device)
    
    model = functions_ML.CombinedModel(NN_model, CNN_model, nn_hidden_dim=8, cnn_hidden_dim=16, output_dim=2).to(device)
    
    # Optimizer
    optimizer_combined = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_combined, T_max=30) # 30 is num_epochs

    # Mixing 32, 16 bit float for speed-up with scaler 
    scaler = torch.cuda.amp.GradScaler()


    # 4. Train and Evaluate the model
    # =================================================================================
    print("Training combined model...")
    reset_seeds(seed)
    losses_train, losses_val, _ , best_val_loss = functions_ML.train_CombinedModel(
        model, combined_train_loader, combined_val_loader, criterion=criterion,
        optimizer=optimizer_combined, num_epochs=30, # Use a fixed large number of epochs
        plot_loss=False, print_loss=False, early_stop=True, patience=5, print_early_stop=True, trial=trial, 
        scaler = scaler, scheduler=None

    )

    # Evaluate on the validation set to get the score for this trial
    y_true_val, y_pred_val, _, extreme_acc_val, nonextreme_acc_val = functions_ML.evaluate_CombinedModel(
        CombinedModel=model, cnn=cnn_model, nn=nn_model,
        test_loader=combined_val_loader, # <-- IMPORTANT: Use validation loader
        print_accuracies=False, train_alone=False
    )
    
    final_val_loss = best_val_loss # use the best val loos found before ealy stoping 


    # =================================================================================
    # 5. Define and Return the Score to be Optimized
    # =================================================================================
    # We want to maximize both accuracies and minimize the loss.

    balanced_accuracy = balanced_accuracy_score(y_true_val, y_pred_val)

    score = balanced_accuracy - final_val_loss
    

    return score


#==========================================================================================================================================================
# EXECUTION BLOCK 
#==========================================================================================================================================================

sites = ['cordoba', 'stockholm', 'hannover', 'lyon', 'belgrado']

best_params_per_site = {}
seed = list_seeds[0] # Using a single seed for the tuning process

# Load the train features for ERA5, reducing time 

train_features_era5 = functions_ML.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TRAIN_DATASET_CONF)

# Start hyperparameter tuning for each site

for site in sites:

    if spei_spi == 'spi':
        files_spei =[f"/spi_data.nc"
                    for scale_spei in scales_spei] # For testing with 1 month scale
    elif spei_spi == 'spei':
        files_spei = [f"/spei_data.nc"
                        for scale_spei in scales_spei]

    spei_spi_variable_mapping = {
    'spei': [f'spei_hg_{scale_spei}' for scale_spei in scales_spei],
    'spi': [f'spi_{scale_spei}' for scale_spei in scales_spei]
    }
    
    spei_variables = spei_spi_variable_mapping[spei_spi] # Variable name in the dataset for SPEI

    train_dataset = ERA5LandDataset_extremes_location_spei(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{percentile_to_load}_{site}_lagged_standarized_anomalies_and_extreme_detection.nc", file_CO2=file_CO2 , files_spei = files_spei, **_ERA5LAND_TRAIN_DATASET_CONF, spei_variables = spei_variables, num_lags=7)
    
    combined_train_dataset = functions_ML.CombinedDataset(train_dataset, train_features_era5,variables = ['g500', 'g200', 'psl'] )
    
    g = torch.Generator()
    g.manual_seed(seed)
    reset_seeds(seed)

    train_size_combined = int(0.8 * len(combined_train_dataset))
    val_size_combined = len(combined_train_dataset) - train_size_combined
    train_subset_combined, val_subset_combined = random_split(combined_train_dataset, [train_size_combined, val_size_combined], generator=g)

    print(f"\n--- Starting Hyperparameter Tuning for site: {site} ---")

    # The lambda function is used to pass extra arguments (site, seed) to the objective function
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)) # Maximize the score defined!
    study.optimize(lambda trial: objective(trial, site=site, seed=seed, 
                    train_subset_combined=train_subset_combined , val_subset_combined= val_subset_combined), n_trials=10) # Run n trials

    # Store the best parameters found for the site
    best_params = study.best_trial.params
    best_params_per_site[site] = best_params

    print(f"--- Tuning Finished for {site} ---")

    best_trial = study.best_trial
    
    print(f"Best score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    results_content = f"Best hyperparameters for site: {site}\n"
    results_content += f"Best trial score: {best_trial.value:.4f}\n\n"
    for key, value in best_trial.params.items():
        if isinstance(value, (float, np.floating)):
            line = f"{key}: {value:.6f}\n"
        else:
            line = f"{key}: {value}\n"
        print(f"    {line.strip()}")
        results_content += line
        
    # Save the best parameters to a text file
    output_dir = "/your/directory/to/save/the/results"
    file_path = os.path.join(output_dir, f"file_name.txt")
    #with open(file_path, 'w') as f:
    #    f.write(results_content)

# Now you have the best hyperparameters for each site
print("\n--- All studies complete ---")
print("Best hyperparameters found for each site:")
print(best_params_per_site)



