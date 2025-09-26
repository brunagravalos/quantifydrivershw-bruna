
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
import gc
import tqdm
import argparse

import functions_ML
from functions_ML import convnext_functions
from functions_ML import ERA5LandDataset_extremes_location_spei


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


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# ======================================================================================================
# DEFINE FUNCTIONS: seed treatment, loading of hyperparameters from hypm optimization 
# ======================================================================================================

def check_seeds():
    print(f"Torch seed: {torch.initial_seed()}")
    print(f"NumPy seed: {np.random.get_state()[1][0]}")
    print(f"Python random seed: {random.getstate()[1][0]}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")

def verify_determinism():
    # Check PyTorch
    print(f"PyTorch rand(): {torch.rand(1).item()}")  # Should match across runs
    # Check NumPy
    print(f"NumPy rand(): {np.random.rand()}")  # Should match
    # Check Python random
    print(f"Python random(): {random.random()}")  # Should match

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)  # For DataLoader's generator

def generate_ensemble_seeds(fixed_seed=123):
    rng = np.random.default_rng(fixed_seed) #generator
    seeds = rng.integers(low=0, high=2**32 - 1, size=20).tolist()
    return seeds

def load_hypms_from_file(site_name, percentile='90p', base_path='/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs',file_name=None):
    """
    Loads hyperparameters for a given site and percentile from a text file. The hyperparameters to load are hardcoded. 
    
    Args:
        site_name (str): The name of the site (e.g., 'cordoba').
        percentile (str): The percentile string, e.g., '95p' or '98p'.
        base_path (str): The directory containing the hyperparameter files.
        file_name (str): The name of the hyperparameter file. If None, it defaults to a standard naming convention.
        
    Returns:
        dict: A dictionary with the loaded hyperparameters or None if the file doesn't exist.
    """
    hypms = {}
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None

    # This mapping handles differences between keys in the file and keys in your script's code.
    key_mapping = {
        'extreme_weights_ctt': 'extreme_weights_ctt',
        'nonextreme_weights_ctt': 'nonextreme_weights_ctt'
    }

    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Use the mapped key if it exists, otherwise use the original key
                code_key = key_mapping.get(key, key)
                
                # Try to convert value to a number, skipping lines where this fails (like headers)
                try:
                    numeric_value = float(value)
                    if code_key == 'batch_size':
                        hypms[code_key] = int(numeric_value)
                    # Only add keys that are expected in the script
                    elif code_key in ['lr', 'w_decay', 'minority_weight_multiplier']:
                         hypms[code_key] = numeric_value
                except ValueError:
                    continue
                    
    return hypms

# ======================================================================================================
# ======================================================================================================

check_seeds()

# Generate list of seeds for the ensamble ---------------------------------------------------------------

list_seeds = generate_ensemble_seeds(fixed_seed=123)

# Get site from bash argument -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train combined model for a specific seed.")
parser.add_argument("seed_value", type=str, help="seed value, member of ensamble")
args = parser.parse_args()
seed_to_process = int(args.seed_value)

# Get seed to process from bash script  
seed = seed_to_process

# Define sites and base hyperparameters ----------------------------------------------------------------------

sites = ['cordoba', 'hannover', 'lyon', 'stockholm', 'belgrado','marrakech']

SITE_HYPMS_fixed = {'belgrado':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1},
             'hannover':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'stockholm':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'lyon':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'cordoba':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'marrakech':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}}

# Create empty dictionary with the base HYPMS
SITE_HYPMS = SITE_HYPMS_fixed.copy()

print(f"Loading hyperparameters for percentile: {percentile_to_load}")
for site_name in sites:
    params = load_hypms_from_file(site_name, percentile=percentile_to_load,file_name = "file_with_hypms.txt")
    if params:
        SITE_HYPMS[site_name] = params #Modify if you want to read HYPS from file 
    print(f"Loaded hyperparameters for {site_name}: {SITE_HYPMS[site_name]}")

# ==============================================================================================================
# Choose which percentile's hyperparameters to load
percentile_to_load = '90p' 
# ==============================================================================================================
# Start loop for sites ------------------------------------------------------------------------------------------------
for site in sites: 

# -----------------------------------------------------------------------------------------------------------
        
    print(f"Doing site: {site}")

    #File paths ERA5 data -------------------------------------------------------------------------------------------
            
    file_g500 = "/path/to/your/data/g500_1x1_lagged_standarized_anomalies.nc"
    file_g200 = "/path/to/your/data/g200_1x1_lagged_standarized_anomalies.nc"
    file_psl = "/path/to/your/data/psl_1x1_lagged_standarized_anomalies.nc"

    # File local scale data and extreme classification ------------------------------------------------
    file_local_scale = "/path/to/your/data/lagged_standarized_anomalies_and_extreme_detection.nc"
    
    # File CO2 data 
    
    file_CO2 = "/path/to/your/data/CO_data.nc"

    # =================================================================================================================
    # Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
    # =================================================================================================================

    HYPMS = dict(
        epochs= 75,
        lr= SITE_HYPMS[site]['lr'],
        w_decay= SITE_HYPMS[site]['w_decay'],
    )
        
    # Datasets configuration dictionaries --------------------------------------------------

    # Start date for all datasets
    start_date = "1950-01-01"
    # Variables large-scale and local scale to use 
    variables_era5 = ['g500','g200','psl']  # g500, g200, psl
    variables_era5land = ['swvl1','swvl2','swvl3']  # swvl1, swvl2, swvl3
    
    # Local-scale datasets configuration
    _ERA5LAND_TRAIN_DATASET_CONF = dict(
    start_date= start_date,
    end_date= "2013-12-31",
    months = [6,7,8],
    variables = variables_era5land
        )

    _ERA5LAND_TEST_DATASET_CONF = dict(
    start_date= "2014-01-01",
    end_date= "2023-12-31",
    months = [6,7,8],
    variables = variables_era5land
        )

    # Large-scale datasets configuration
    _ERA5_TRAIN_DATASET_CONF = dict(
    start_date= start_date,
    end_date= "2013-12-31",
    months = [6,7,8],
    start_lag = 1,
    lags_era5 = 1,
    variables = variables_era5
        )

    _ERA5_TEST_DATASET_CONF = dict(
    start_date= "2014-01-01",
    end_date= "2023-12-31",
    months = [6,7,8],
    start_lag = 1,
    lags_era5 = 1,
    variables = variables_era5      
        )

    # Number of lags large-scale fields
    number_lags = _ERA5_TEST_DATASET_CONF['variables']

    # Name to save the trained CombinedModel
    name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"

    
    # Datasets ERA5land data --------------------------------------------------------------
    
    train_dataset = functions_ML.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=file_local_scale, file_CO2=file_CO2 , **_ERA5LAND_TRAIN_DATASET_CONF)
    test_dataset = functions_ML.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=file_local_scale, file_CO2=file_CO2 , **_ERA5LAND_TEST_DATASET_CONF)
    
    # Datasets ERA5 data ------------------------------------------------------------------

    train_features_era5 = functions_ML.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TRAIN_DATASET_CONF) # shape: features, time, lat, lon 
    test_features_era5 = functions_ML.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TEST_DATASET_CONF)
    
    # =========================================================================================
    # Dataloaders configuration dictionaries --------------------------------------------------
    # =========================================================================================

    g = torch.Generator()
    reset_seeds(seed)


    _DATALOADERS_CONF = dict(
    batch_size= SITE_HYPMS[site]['batch_size'],
    drop_last= False,
    shuffle = True,
    num_workers=0,
    generator=g       
        )
    
    _DATALOADERS_TEST_CONF = dict(
            batch_size= SITE_HYPMS[site]['batch_size'],
            drop_last= False,
            shuffle = False,
            num_workers=0
        )

    # =========================================================================================
    
    # Combined Dataset and Dataloader 
    
    batch_size = _DATALOADERS_CONF['batch_size'] # batch size for dataloaders both datasets
    
    combined_train_dataset  = functions_ML.CombinedDataset(train_dataset,train_features_era5, variables=variables_era5)
    combined_test_dataset = functions_ML.CombinedDataset(test_dataset,test_features_era5, variables=variables_era5)
    
    # Split train and validation sets for the combined dataset ------------------------------------------------
    train_size_combined = int(0.8 * len(combined_train_dataset))
    val_size_combined = len(combined_train_dataset) - train_size_combined
    
    #random split 
    train_subset_combined, val_subset_combined = random_split(combined_train_dataset, [train_size_combined, val_size_combined],generator=g)
    
    # (local,regional,labels)
    combined_train_loader = DataLoader(train_subset_combined, **_DATALOADERS_CONF) 
    combined_val_loader = DataLoader(val_subset_combined, **_DATALOADERS_CONF)
    combined_test_loader = DataLoader(combined_test_dataset, **_DATALOADERS_TEST_CONF)
    
    
    #  Weights class imbalance  ---------------------------------------------------------------------------------
    
    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    total_counts = sum(class_counts)
    
    # Alternative way to compute class weights, if wanted to use weights for each class -------------------------

    #extreme_weights_ctt = SITE_HYPMS[site]['extreme_weights_ctt']
    #nonextreme_weights_ctt = SITE_HYPMS[site]['nonextreme_weights_ctt']
    #class_weights = torch.tensor([total_counts / (nonextreme_weights_ctt*class_counts[0]), total_counts / (extreme_weights_ctt*class_counts[1])], dtype=torch.float)
    # -------------------------------------------------------------------------------------------------------------

    base_minority_weight = class_counts[0] / class_counts[1]

    minority_weight_multiplier = SITE_HYPMS[site]['minority_weight_multiplier']
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device) # smoothing the weights
        
    # Cross-entropy loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=smoothed_weights) 

    
    # Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
    
    reset_seeds(seed)
    # MLP for local-scale
    NN_model = functions_ML.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    reset_seeds(seed)
    # ConvNext for large-scale
    CNN_model_loaded = convnext_functions.ConvNext(
           num_channels=len(train_features_era5.all_features),
           num_classes=2,
           patch_size=4,
           layer_dims=[4, 6,6,16],
           depths=[1, 2,2,1],
           drop_rate=0.05,
           train_alone=False,
    ).to(device) 
    reset_seeds(seed)
    
    # Combined model    
    model = functions_ML.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(seed)
    
    # =======================================================================================================================================
    # Train phase Combined model -------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================
    
    reset_seeds(seed)
    # Optimizer 
    optimizer_combined = optim.AdamW(model.parameters(), lr=HYPMS['lr'], weight_decay=HYPMS['w_decay'])  
    # Scheduler (if wanted)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_combined, T_max=30) 

    # Start training
    print( " Training combined model " )
    losses_train_combined, losses_val_combined, num_e, best_val_loss = functions_ML.train_CombinedModel(model,combined_train_loader, combined_val_loader, criterion=criterion,
                                                                                                               optimizer=optimizer_combined,num_epochs=HYPMS['epochs'],
                                                                                                               plot_loss=False,print_loss=False, early_stop=True, patience=5,
                                                                                                               print_early_stop=False,trial=None,scaler=None,scheduler=None)
    
    # Save the trained CombinedModel -----------------------------
    torch.save(model.state_dict(), "/your/path/to/save/weights/model/file_name.pth")
    
    # =======================================================================================================================================
    # Evaluation phase ----------------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================

    reset_seeds(seed)
    y_true, y_pred, outputs_prob,extreme_acc, nonextreme_acc = functions_ML.evaluate_CombinedModel(CombinedModel=model,cnn=cnn_model_loaded,nn=nn_model_loaded, test_loader=combined_test_loader, print_accuracies=True,train_alone=False)
    reset_seeds(seed)
    
    # Save the dictionaries with the relevant data ---------------------------------------------------------------------
    seed_results = {
            'y_true_pred_pairs': (y_true,y_pred), # These are from the current site
            'out_probs_seed': outputs_prob,
            'extreme_accuracy': extreme_acc,
            'nonextreme_accuracy': nonextreme_acc,
            'losses_train': losses_train_combined,
            'losses_val': losses_val_combined
        }
        
    main_path = '/your/path/to/save/results'  # Change to your desired path
    
    with open(os.path.join(main_path, f'{site}/{percentile_to_load}_results_data_{seed}.pkl'), 'wb') as f:
        pickle.dump(seed_results, f)
        print(f"saved file results {seed}")
    
    print("Finished training model, computing SHAP")
    

    # =======================================================================================================================================
    # SHAP computation ----------------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================
    
    # Prepare NN model and CNN model for SHAP -----------------------------------------------------------------------------------------------
    NN_model_loaded = functions_ML.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    NN_model_loaded.eval()
    reset_seeds(seed)
    CNN_model_loaded = convnext_functions.ConvNext(
           num_channels=len(train_features_era5.all_features),
           num_classes=2,
           patch_size=4,
           layer_dims=[4,6,6,16],
           depths=[1, 2,2,1],
           drop_rate=0.05,
           train_alone=False,
    ).to(device) 
    reset_seeds(seed)
    
    # Create the Combined model for SHAP---------------------------------------------------------------------------------------------------
    model = functions_ML.CombinedModel(NN_model_loaded, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(seed)
    # Load the trained CombinedModel weights ----------------------------------------------------------------------------------------------
    model_state_dict = torch.load("/your/path/to/save/weights/model/file_name.pth")
    model.load_state_dict(model_state_dict)
    model.eval()

    # ---------------------------------------------------------------------------------------------------------------------
    
    # Create baseline for SHAP computation --------------------------------------------------------------------------------
    background_indices = np.random.choice(len(train_subset_combined), 200, replace=False)
    background_nn = []
    background_cnn = []
    
    for idx in background_indices:
        nn_input, cnn_input, labels = train_subset_combined[idx]  # Adjust based on your dataset structure
        background_nn.append(nn_input)
        background_cnn.append(cnn_input)
    
    device = next(model.parameters()).device
    
    background_nn = torch.stack(background_nn,dim=0).to(device)
    background_cnn = torch.stack(background_cnn,dim=0).to(device)
    
    # Create explainer dataset for SHAP computation -----------------------------------------------------------------------
    explainer_indices = np.arange(0,len(combined_test_dataset),1)
    explain_nn = []
    explain_cnn = []
    
    for idx in explainer_indices:
        nn_input, cnn_input, labels = combined_test_dataset[idx]  # Adjust based on your dataset structure
        explain_nn.append(nn_input)
        explain_cnn.append(cnn_input)
    
    explain_nn = torch.stack(explain_nn,dim=0).to(device)
    explain_cnn = torch.stack(explain_cnn,dim=0).to(device)
    
    # Merge local-scale and large-scale data for SHAP computation ------------------------------------------------------
    background_data = [background_nn, background_cnn]
    explain_data = [explain_nn, explain_cnn]
    
    reset_seeds(seed)
    print("Initializing GradientExplainer...")
    explainer_grad = shap.GradientExplainer(model, background_data)
    print("Explainer initialized.")
    reset_seeds(seed)
    print("Calculating SHAP values...")
    shap_values = explainer_grad.shap_values(explain_data) # Still check additivity
    
    print(f"Finished computing SHAP values for site: {site}")
    
    # Select class to explaine, extreme (1) in our case ----------------------------------------------------------------
    class_index_to_explain = 1
    shap_values_nn_raw = shap_values[0][:,:,1] # NumPy array (N_explain, nn_features)
    shap_values_cnn_raw = shap_values[1][:,:,:,:,1] # NumPy array (N_explain, V, H, W)
    
    # Dictionary to save SHAP values -----------------------------------------------------------------------------------
    raw_shap_dict = {
        'nn': shap_values_nn_raw,
        'cnn': shap_values_cnn_raw,
    }
    
    with open(f'/your/path/to/save/SHAP/results', 'wb') as f:

        pickle.dump(raw_shap_dict, f)
    
    print(f"Finished training and SHAP value computing for site: {site}")
    
