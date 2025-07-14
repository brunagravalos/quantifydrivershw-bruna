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
import functions_NN_extremes
import torch.nn as nn                   # provides classes and functions to create and train neural networks
import torch.nn.functional as F         # provides functions for activation functions, loss functions, and other operations
import functions_NN_extremes
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re 
import shap 
from sklearn.metrics import balanced_accuracy_score
import pickle
import functions_improve_CombinedModel
import convnext_functions
import modified_convnext_functions
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

import gc
import tqdm
import argparse

# ==================================================================================================================================
# 1. Global Configuration
#===================================================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Site to be done  *************************

# Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------

HYPMS = dict(
    epochs= 75,
    lr= 1e-4,
    w_decay= 0.01,
)

# Datasets configuration --------------------------------------------------------------------------------------------------------------------------------------------------

start_date = "1950-01-01"
    
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

_ERA5_TRAIN_DATASET_CONF = dict(
start_date= start_date,
end_date= "2013-12-31",
start_lag = 1,
lags_era5 = 3,
months = [6,7,8],
    )

_ERA5_TEST_DATASET_CONF = dict(
start_date= "2014-01-01",
end_date= "2023-12-31",
start_lag = 1,
lags_era5 = 3,
months = [6,7,8]         
    )

# Number of lags for the larg-scale data ----------------------------------------------------------------------------------------------------------------------------------

number_lags = 3

#File paths ERA5 data -----------------------------------------------------------------------------------------------------------------------------------------------------
        
file_g500 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/g500_1x1_lagged_standarized_anomalies.nc"
file_g200 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/g200_1x1_lagged_standarized_anomalies.nc"
file_psl = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/psl_1x1_lagged_standarized_anomalies.nc"
file_hus850 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus850_lagged_standarized_anomalies.nc"
file_hus975 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus975_lagged_standarized_anomalies.nc"
file_rsds = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/rsds_lagged_standarized_anomalies.nc"
file_hus700 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus700_lagged_standarized_anomalies.nc"

# File CO2 data -----------------------------------------------------------------------------------------------------------------------------------------------------------

file_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"

# File local scale data ---------------------------------------------------------------------------------------------------------------------------------------------------

# Defined inside the objective function


# Train alone configuration ----------------------------------------------------------------------------------------------------------------------------------------------

# Flags to train CNN and NN separated or not *************************

train_nn_alone = False
train_cnn_alone = False


# Names to save trained models -------------------------------------------------------------------------------------------------------------------------------------------

name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
name_save_CNN = "cnn_trained_alone"
name_save_NN = "nn_trained_alone"
name_save_losses_fig = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"

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
    # Suggest weights for the loss function
    extreme_weights_ctt = trial.suggest_float("extreme_weights_ctt", 0.5, 2.0)
    nonextreme_weight_ctt = trial.suggest_float("nonextreme_weight_ctt", 0.5, 10.0)
    
    print(f"Trial {trial.number} for site {site}: batch_size={batch_size}, lr={lr:.6f}, w_decay={w_decay:.6f}, extreme_weight={extreme_weights_ctt:.2f}")

    reset_seeds(seed)

    # 2. Setup Dataloaders

    g = torch.Generator()
    g.manual_seed(seed)

    _DATALOADERS_CONF = dict(
        batch_size=batch_size, 
        drop_last=False,
        shuffle=True,
        num_workers=4,
        generator=g
    )
    _DATALOADERS_VAL_CONF = dict(
        batch_size=batch_size, 
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )   

    reset_seeds(seed)

    combined_train_loader = DataLoader(train_subset_combined, **_DATALOADERS_CONF)
    combined_val_loader = DataLoader(val_subset_combined, **_DATALOADERS_VAL_CONF)

    # 3. Setup Model, Optimizer and Loss with suggested hyperparameters
    # =================================================================================
    # Class weights
    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    class_weights = torch.tensor([1. / (nonextreme_weight_ctt * class_counts[0]), 1. / (extreme_weights_ctt * class_counts[1])], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize models
    reset_seeds(seed)
    nn_model = functions_NN_extremes.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    cnn_model = convnext_functions.ConvNext(
        num_channels=len(train_features_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False
    ).to(device)
    
    model = functions_NN_extremes.CombinedModel(nn_model, cnn_model, nn_hidden_dim=8, cnn_hidden_dim=16, output_dim=2).to(device)
    
    # Optimizer
    optimizer_combined = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)

    # 4. Train and Evaluate the model
    # =================================================================================
    print("Training combined model...")
    reset_seeds(seed)
    losses_train, losses_val, _ , best_val_loss = functions_NN_extremes.train_CombinedModel(
        model, combined_train_loader, combined_val_loader, criterion=criterion,
        optimizer=optimizer_combined, num_epochs=HYPMS['epochs'], # Use a fixed large number of epochs
        plot_loss=False, print_loss=False, early_stop=True, patience=5, print_early_stop=True, trial=trial
    )

    # Evaluate on the validation set to get the score for this trial
    y_true_val, y_pred_val, _, extreme_acc_val, nonextreme_acc_val = functions_NN_extremes.evaluate_CombinedModel(
        CombinedModel=model, cnn=cnn_model, nn=nn_model,
        test_loader=combined_val_loader, # <-- IMPORTANT: Use validation loader
        print_accuracies=False, train_alone=False
    )
    
    final_val_loss = best_val_loss # use the best val loos found before ealy stoping 

    # 5. Define and Return the Score to be Optimized
    # =================================================================================
    # We want to maximize both accuracies and minimize the loss.
    # A good composite score could be the balanced accuracy, penalized by the loss.
    #balanced_accuracy = (extreme_acc_val + nonextreme_acc_val) / 2.0
    balanced_accuracy = balanced_accuracy_score(y_true_val, y_pred_val)
    
    # We subtract the loss because Optuna can only either maximize or minimize.
    # By maximizing (balanced_accuracy - final_val_loss), we achieve both goals.
    score = balanced_accuracy - final_val_loss + (0.1 * extreme_acc_val / 100) + (0.1 * nonextreme_acc_val / 100)

    return score


#==========================================================================================================================================================
# EXECUTION BLOCK 
#==========================================================================================================================================================

sites = ['cordoba', 'stockholm', 'hannover', 'lyon', 'belgrado', 'marrakech']

sites = ['cordoba']

best_params_per_site = {}
seed = list_seeds[0] # Using a single seed for the tuning process

for site in sites:

    # Prepare datasets 
    file_local_scale = f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/std_changed_{site}_lagged_standarized_anomalies_and_95p_extreme_detection.nc"
    
    train_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=file_local_scale, file_CO2=file_CO2 ,**_ERA5LAND_TRAIN_DATASET_CONF)
    train_features_era5 = functions_NN_extremes.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TRAIN_DATASET_CONF)
    
    combined_train_dataset = functions_NN_extremes.CombinedDataset(train_dataset, train_features_era5)
    
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
                    train_subset_combined=train_subset_combined , val_subset_combined= val_subset_combined), n_trials=20) # Run 20 trials

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
        line = f"{key}: {value}\n"
        print(f"    {line.strip()}")
        results_content += line
        
    # Save the best parameters to a text file
    output_dir = "/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs/"
    file_path = os.path.join(output_dir, f"{site}_best_params_95p_with_testing_phase.txt")
    with open(file_path, 'w') as f:
        f.write(results_content)

# Now you have the best hyperparameters for each site
print("\n--- All studies complete ---")
print("Best hyperparameters found for each site:")
print(best_params_per_site)



