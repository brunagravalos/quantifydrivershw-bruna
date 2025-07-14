import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

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


# Generate list of seeds for the ensamble ---------------------------------------------------------------

list_seeds = generate_ensemble_seeds(fixed_seed=123)

# -------------------------------------------------------------------------------------------------------


raw_shap_dictionaries = {'cordoba':[],'lyon':[],'hannover':[],'stockholm':[],'belgrado':[],'marrakech':[]}

sites = ['cordoba','lyon','hannover','stockholm','belgrado','marrakech']

sites_title = ['Córdoba', 'Lyon', 'Hannover', 'Stockholm', 'Belgrade', 'Marrakech']

number_lags = 3

name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"

#File paths ERA5 data 
        
file_g500 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/g500_1x1_lagged_standarized_anomalies.nc"
file_g200 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/g200_1x1_lagged_standarized_anomalies.nc"
file_psl = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/psl_1x1_lagged_standarized_anomalies.nc"
file_hus850 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus850_lagged_standarized_anomalies.nc"
file_hus975 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus975_lagged_standarized_anomalies.nc"
file_rsds = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/rsds_lagged_standarized_anomalies.nc"
file_hus700 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus700_lagged_standarized_anomalies.nc"



# File CO2 data 

file_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"

# Start loop for sites --------------------------------------------------------------------


count_plot = 0

sites = ['stockholm']


for site in sites: 

    print(f"Doing site: {site}")

    # Datasets ERA5land data 

    if site == 'belgrado': # Special case Belgrado to capture correctly the trend, discarding special case 1950 and 1952, which have a high extreme count. 
            start_date = "1953-01-01"
    else:
        start_date = "1950-01-01"
    
    train_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{site}_lagged_standarized_anomalies_and_extreme_detection.nc", file_CO2=file_CO2 ,start_date=start_date, end_date="2013-12-31", months=[6,7,8])
    test_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{site}_lagged_standarized_anomalies_and_extreme_detection.nc", file_CO2=file_CO2 ,start_date="2014-01-01", end_date="2024-12-31", months=[6,7,8])

    #train_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{site}_lagged_standarized_anomalies_and_extreme_detection.nc", start_date="1950-01-03", end_date="2013-12-31", months=[6,7,8])
    #test_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/{site}_lagged_standarized_anomalies_and_extreme_detection.nc", start_date="2014-01-01", end_date="2024-12-31", months=[6,7,8])

    # Train features ERA5
        
    #train_features_era5 = functions_NN_extremes.ERA5Dataset_extremes_single_field(file_g500,'g500', start_date='1950-01-03', end_date='2013-12-31',months=[6,7,8],lags_era5=number_lags+1) # shape: features, time, lat, lon 
    #test_features_era5 = functions_NN_extremes.ERA5Dataset_extremes_single_field(file_g500,'g500', start_date='2014-01-01', end_date='2024-12-31', months=[6,7,8],lags_era5=number_lags+1)
    
    train_features_era5 = functions_NN_extremes.ERA5Dataset_extremes(file_g500,file_g200,file_psl, start_date=start_date, end_date='2013-12-31',months=[6,7,8],lags_era5=number_lags+1) # shape: features, time, lat, lon 
    test_features_era5 = functions_NN_extremes.ERA5Dataset_extremes(file_g500,file_g200,file_psl, start_date='2014-01-01', end_date='2024-12-31', months=[6,7,8],lags_era5=number_lags+1)

    ensamble_shap_values_nn_raw = np.zeros((len(list_seeds),test_dataset.features.shape[0],test_dataset.features.shape[1]))
    ensamble_shap_values_cnn_raw = np.zeros((len(list_seeds),test_dataset.features.shape[0],test_features_era5.features.shape[1],58,125))

    for count_seed,seed in enumerate(list_seeds): # Loop over seeds for the ensamble 

        print(f"Doing member: {count_seed+1}")

        g = torch.Generator()
        reset_seeds(seed)
          
        _DATALOADERS_CONF = dict(
        batch_size= 32,
        drop_last= False,
        shuffle = True,
        num_workers=0,
        generator=g        # make shuffle deterministic 
            )
        
        _DATALOADERS_TEST_CONF = dict(
                batch_size= 32,
                drop_last= False,
                shuffle = False,
                num_workers=0
            )


        outputs_prob_seeds = np.zeros((len(list_seeds),test_dataset.features.shape[0],2))
        
        # Combined Dataset and Dataloader 
    
        batch_size = _DATALOADERS_CONF['batch_size'] # batch size for dataloaders both datasets
        
        #combined_train_dataset  = functions_NN_extremes.CombinedDataset_single_field(train_dataset,train_features_era5)
        #combined_test_dataset = functions_NN_extremes.CombinedDataset_single_field(test_dataset,test_features_era5)
    
        combined_train_dataset  = functions_NN_extremes.CombinedDataset(train_dataset,train_features_era5)
        combined_test_dataset = functions_NN_extremes.CombinedDataset(test_dataset,test_features_era5)
        
        
        train_size_combined = int(0.8 * len(combined_train_dataset))
        val_size_combined = len(combined_train_dataset) - train_size_combined
        
        #random split 
        train_subset_combined, val_subset_combined = random_split(combined_train_dataset, [train_size_combined, val_size_combined],generator=g)
    
        # (local,regional,labels)
        combined_train_loader = DataLoader(train_subset_combined, **_DATALOADERS_CONF) # When using DataLoader with shuffle=True, PyTorch randomly picks idx values.
        combined_val_loader = DataLoader(val_subset_combined, **_DATALOADERS_CONF)
        combined_test_loader = DataLoader(combined_test_dataset, **_DATALOADERS_TEST_CONF)
        
        reset_seeds(seed)
    
        nn_model_loaded = functions_NN_extremes.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
        nn_model_loaded.eval()
        reset_seeds(seed)
        cnn_model_loaded = convnext_functions.ConvNext(
            num_channels=len(train_features_era5.all_features),
            num_classes=2,
            patch_size=2,
            layer_dims=[4, 6,6,16],
            depths=[1, 2,2,1],
            drop_rate=0.05,
            train_alone=False
            #combined_hidden_dim=64
        ).to(device) 
        cnn_model_loaded.eval()
        reset_seeds(seed)
        
        model = functions_NN_extremes.CombinedModel(nn_model_loaded, cnn_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
        reset_seeds(seed)
        model_state_dict = torch.load(f"trained_models/{site}/padding_test/member_{count_seed+1}_{name_save_CombinedModel}_{site}_test.pth")
        model.load_state_dict(model_state_dict)
        model.eval()



        background_indices = np.random.choice(len(train_subset_combined), 100, replace=False)
        background_nn = []
        background_cnn = []
        
        for idx in background_indices:
            nn_input, cnn_input, labels = train_subset_combined[idx]  # Adjust based on your dataset structure
            background_nn.append(nn_input)
            background_cnn.append(cnn_input)
    
        #background_nn = nn_input.to(device)
        #background_cnn = cnn_input.to(device)
        
        device = next(model.parameters()).device
    
        background_nn = torch.stack(background_nn,dim=0).to(device)
        background_cnn = torch.stack(background_cnn,dim=0).to(device)
        
        explainer_indices = np.arange(0,len(combined_test_dataset),1)
        explain_nn = []
        explain_cnn = []
    
        for idx in explainer_indices:
            nn_input, cnn_input, labels = combined_test_dataset[idx]  # Adjust based on your dataset structure
            explain_nn.append(nn_input)
            explain_cnn.append(cnn_input)
    
        explain_nn = torch.stack(explain_nn,dim=0).to(device)
        explain_cnn = torch.stack(explain_cnn,dim=0).to(device)
        
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
        
    
        class_index_to_explain = 1
        shap_values_nn_raw = shap_values[0][:,:,1] # NumPy array (N_explain, nn_features)
        shap_values_cnn_raw = shap_values[1][:,:,:,:,1] # NumPy array (N_explain, V, H, W)

        ensamble_shap_values_nn_raw[count_seed] = shap_values_nn_raw
        ensamble_shap_values_cnn_raw[count_seed] = shap_values_cnn_raw

    mean_ensamble_shap_values_nn_raw = np.mean(ensamble_shap_values_nn_raw,axis=0)
    mean_ensamble_shap_values_cnn_raw = np.mean(ensamble_shap_values_cnn_raw,axis=0)

    std_ensamble_shap_values_nn_raw = np.std(ensamble_shap_values_nn_raw,axis=0)
    std_ensamble_shap_values_cnn_raw = np.std(ensamble_shap_values_cnn_raw,axis=0)
    
    raw_shap_dict = {
        'nn_mean': mean_ensamble_shap_values_nn_raw,
        'nn_std': std_ensamble_shap_values_nn_raw,
        'cnn_mean': mean_ensamble_shap_values_cnn_raw,
        'cnn_std': std_ensamble_shap_values_cnn_raw,
    }

    with open(f'SHAP_values_Gradient/padding_test_SHAP_values_GradientExplainer_not_filtered_{site}_test_{number_lags}lags.pkl', 'wb') as f:
        pickle.dump(raw_shap_dict, f)

    print(f"Finished training and SHAP value computing for site: {site}")