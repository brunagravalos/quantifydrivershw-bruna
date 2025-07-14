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
import convnext_functions_modifications
from dependent_gradient import DependentGradientExplainer
from conditional_sampler import GaussianCopulaSampler

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


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

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


# Generate list of seeds for the ensamble ---------------------------------------------------------------

list_seeds = generate_ensemble_seeds(fixed_seed=123)

# Get site from bash argument -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train combined model for a specific seed.")
#parser.add_argument("site_name", type=str, help="The name of the site to process (e.g., 'cordoba')")
parser.add_argument("seed_value", type=str, help="seed value, member of ensamble")
args = parser.parse_args()
seed_to_process = int(args.seed_value)

sites = ['cordoba', 'stockholm', 'lyon', 'marrakech']

sites = ['cordoba']

SITE_HYPMS_95p = {'belgrado':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8},
             'hannover':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8}, 
             'stockholm':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8}, 
             'lyon':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8}, 
             'cordoba':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8}, 
             'marrakech':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':0.8}}

SITE_HYPMS_98p = {'belgrado':{'lr':0.000316,'w_decay':0.00073,'batch_size':32,'extreme_weight_ctt':7.568,'nonextreme_weight_ctt':1.487},
             'hannover':{'lr':0.0002,'w_decay':0.00015,'batch_size':16,'extreme_weight_ctt':9.953059,'nonextreme_weight_ctt':1.152219}, 
             'stockholm':{'lr':0.000355,'w_decay':0.0134,'batch_size':32,'extreme_weight_ctt':9.5268,'nonextreme_weight_ctt':0.8913}, 
             'lyon':{'lr':0.00034,'w_decay':0.05465,'batch_size':16,'extreme_weight_ctt':9.56628,'nonextreme_weight_ctt':0.506}, 
             'cordoba':{'lr':0.000219,'w_decay':0.09097,'batch_size':32,'extreme_weight_ctt':7.8,'nonextreme_weight_ctt':1.1156}, 
             'marrakech':{'lr':0.000276,'w_decay':2.12e-05,'batch_size':16,'extreme_weight_ctt':8.53106,'nonextreme_weight_ctt':0.671}}

SITE_HYPMS = {'belgrado':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1},
             'hannover':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1}, 
             'stockholm':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1}, 
             'lyon':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1}, 
             'cordoba':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1.}, 
             'marrakech':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weight_ctt':1,'nonextreme_weight_ctt':1}}


for site in sites: 

    # -----------------------------------------------------------------------------------------------------------
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
    
            
    HYPMS = dict(
        epochs= 75,
        lr= SITE_HYPMS[site]['lr'],
        w_decay= SITE_HYPMS[site]['w_decay'],
    )
    
    
    # Flags to train CNN and NN separated or not *************************
    
    train_nn_alone = False
    train_cnn_alone = False
    
    # ********************************************************************
    
    
    # Names to save trained models 
    
    number_lags = 3
    
    name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
    name_save_CNN = "cnn_trained_alone"
    name_save_NN = "nn_trained_alone"
    name_save_losses_fig = f"Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
    
    
    #File paths ERA5 data 
            
    file_g500 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_g500_1x1_lagged_standarized_anomalies.nc"
    file_g200 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_g200_1x1_lagged_standarized_anomalies.nc"
    file_psl = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/std_changed_psl_1x1_lagged_standarized_anomalies.nc"
    file_hus850 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus850_lagged_standarized_anomalies.nc"
    file_hus975 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus975_lagged_standarized_anomalies.nc"
    file_rsds = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/rsds_lagged_standarized_anomalies.nc"
    file_hus700 = "/gpfs/scratch/bsc32/bsc167965/tfm_data/era5/lagged_anomalies/hus700_lagged_standarized_anomalies.nc"
    
    
    
    # File CO2 data 
    
    file_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"
    
    # Start loop for sites --------------------------------------------------------------------
    
    
    count_plot = 0


    # Datasets configuration dictionaries --------------------------------------------------

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

    # --------------------------------------------------------------------------------------

    
    print(f"Doing site: {site}")
    
    # Datasets ERA5land data --------------------------------------------------------------
    
    
    train_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/std_changed_{site}_lagged_standarized_anomalies_and_95p_extreme_detection.nc", file_CO2=file_CO2 , **_ERA5LAND_TRAIN_DATASET_CONF)
    test_dataset = functions_NN_extremes.ERA5LandDataset_extremes_location_swvl_averaged_including_CO2(file_path=f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/lagged_anomalies_and_event_detection/std_changed_{site}_lagged_standarized_anomalies_and_95p_extreme_detection.nc", file_CO2=file_CO2 , **_ERA5LAND_TEST_DATASET_CONF)
    

    train_features_era5 = functions_NN_extremes.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TRAIN_DATASET_CONF) # shape: features, time, lat, lon 
    test_features_era5 = functions_NN_extremes.ERA5Dataset_extremes(file_g500,file_g200,file_psl, **_ERA5_TEST_DATASET_CONF)
    
    
    # Get seed to process from bash script 
    
    seed = seed_to_process
    
    
    
    g = torch.Generator()
    
    reset_seeds(seed)
     
    _DATALOADERS_CONF = dict(
    batch_size= SITE_HYPMS[site]['batch_size'],
    drop_last= False,
    shuffle = True,
    num_workers=0,
    generator=g        # make shuffle deterministic 
        )
    
    _DATALOADERS_TEST_CONF = dict(
            batch_size= SITE_HYPMS[site]['batch_size'],
            drop_last= False,
            shuffle = False,
            num_workers=0
        )
    
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
    
    
    #  Weights class imbalance  ---------------------------------------------------------------------------------
    
    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    total_counts = sum(class_counts)
    
    #class_weights = torch.tensor([1. / (class_counts[0]), 1. / (class_counts[1])], dtype=torch.float)  # relevance 
    extreme_weights_ctt = SITE_HYPMS[site]['extreme_weight_ctt']
    nonextreme_weight_ctt = SITE_HYPMS[site]['nonextreme_weight_ctt']
    
    class_weights = torch.tensor([total_counts / (nonextreme_weight_ctt*class_counts[0]), total_counts / (extreme_weights_ctt*class_counts[1])], dtype=torch.float)
    
    weight_0_raw = total_counts / class_counts[0]
    weight_1_raw = total_counts / class_counts[1]
    
    # Apply square root to soften
    #class_weights = torch.tensor([
    #    torch.sqrt(torch.tensor(weight_0_raw)),
    #    torch.sqrt(torch.tensor(weight_1_raw))
    #], dtype=torch.float)
    
    # relevance 
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights) # use simple NN criterion, that has weighted labels 
    
    
    # Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    # New nn and cnn, now without alone training
    # Step independent of doing separated training or not 
    reset_seeds(seed)
    nn_model_loaded = functions_NN_extremes.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    #cnn_model_loaded = functions_NN_extremes.CNN_Era5ExtemeClassifer(lat_size=58,lon_size=125,num_variables=3,num_lags=number_lags, train_alone_CNN=False, num_classes=2,pretrain_flag=False).to(device) 
    reset_seeds(seed)
    cnn_model_loaded = convnext_functions.ConvNext(
           num_channels=len(train_features_era5.all_features),
           num_classes=2,
           patch_size=4,
           layer_dims=[4, 6,6,16],
           depths=[1, 2,2,1],
           drop_rate=0.05,
           train_alone=False,
           #dilations_per_stage=[1, 1, 1, 1],  # No dilation in first 2 stages, dilation of 2 in last 2. More context aware with the dilations 
           #block_padding_modes_per_stage=['replicate', 'zeros', 'zeros', 'zeros'],     # padding in ConvNextBlocks
           #downsample_padding_mode='replicate'
    ).to(device) 
    reset_seeds(seed)
    
    
    
    # ************* dimension nn_hidden and cnn_hidden with lluis cnn: nn_hidden_dim=mine, cnn_hidden_dim = 2 *************************************************
    
    model = functions_NN_extremes.CombinedModel(nn_model_loaded, cnn_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(seed)
    
    # Train phase Combined model ------------------------------------------------
    
    print( " Training combined model  " )
    reset_seeds(seed)
    optimizer_combined = optim.AdamW(model.parameters(), lr=HYPMS['lr'], weight_decay=HYPMS['w_decay'])    
    
    losses_train_combined, losses_val_combined, num_e = functions_NN_extremes.train_CombinedModel(model,combined_train_loader, combined_val_loader, criterion=criterion,
                                                                                                               optimizer=optimizer_combined,num_epochs=HYPMS['epochs'],
                                                                                                               plot_loss=False,print_loss=False, early_stop=True, patience=5,
                                                                                                               print_early_stop=False,trial=None)
    
    # Save the trained CombinedModel -----------------------------
    
    torch.save(model.state_dict(), f"/gpfs/scratch/bsc32/bsc167965/tfm_data/test_train_n_shap_dilation/{site}/trained_models/member_{seed}_{name_save_CombinedModel}_{site}_test.pth")
    
    # ------------------------------------------------------------
    
    #axes_combined[count_plot].plot(np.arange(1,num_epochs_early_stop[site]+1,1),losses_train_combined,label='train',color='blue')
    #axes_combined[count_plot].plot(np.arange(1,num_epochs_early_stop[site]+1,1),losses_val_combined,label='val',color='orange')
    
    
    # Evaluation phase --------------------------------------
    
    # If we are doing the train of CNN and NN separated, in the final evaluation step we need to refresh CNN and NN models and give them the weights and bias including the last linear layer+
    # that does the final binary classification. This way, we can obtain the outputs of the models separatedely during the evalutation. 
    
    
    reset_seeds(seed)
    y_true, y_pred, outputs_prob,extreme_acc, nonextreme_acc = functions_NN_extremes.evaluate_CombinedModel(CombinedModel=model,cnn=cnn_model_loaded,nn=nn_model_loaded, test_loader=combined_test_loader, print_accuracies=True,train_alone=False)
    
    
    
    # Ensamble probabilitites mean and std and final prediction --------------------------------------------------------------------------------------------------
        
    
    
    reset_seeds(seed)
    
    # Save the dictionaries with the relevant data ---------------------------------------------------------------------
    
    seed_results = {
            'y_true_pred_pairs': (y_true,y_pred), # These are from the current site
            'out_probs_seed': outputs_prob,
            'extreme_accuracy': extreme_acc,
            'nonextreme_accuracy': nonextreme_acc,
            'losses_train': losses_train_combined,
            'losses_val': losses_val_combined
            # You might also want to save losses_train_all_seeds and losses_val_all_seeds for this site
        }
    
    
    main_path = '/gpfs/scratch/bsc32/bsc167965/tfm_data/test_train_n_shap_dilation/'
    
    
    with open(os.path.join(main_path, f'{site}/80p_results_data_{seed}.pkl'), 'wb') as f:
        pickle.dump(seed_results, f)
        print(f"saved file results {seed}")
    
    print("Finished training model, computing SHAP")
    
    # Show figure losses -----------------------------------------------------------------------------------------------
    
    nn_model_loaded = functions_NN_extremes.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2).to(device)
    nn_model_loaded.eval()
    reset_seeds(seed)
    cnn_model_loaded = convnext_functions.ConvNext(
           num_channels=len(train_features_era5.all_features),
           num_classes=2,
           patch_size=4,
           layer_dims=[4, 6,6,16],
           depths=[1, 2,2,1],
           drop_rate=0.05,
           train_alone=False,
           #dilations_per_stage=[1, 1, 1, 1],  # No dilation in first 2 stages, dilation of 2 in last 2. More context aware with the dilations 
           #block_padding_modes_per_stage=['replicate', 'replicate', 'replicate', 'replicate'],     # padding in ConvNextBlocks
           #downsample_padding_mode='replicate'
    ).to(device) 
    reset_seeds(seed)
    
    model = functions_NN_extremes.CombinedModel(nn_model_loaded, cnn_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(seed)
    model_state_dict = torch.load(f"/gpfs/scratch/bsc32/bsc167965/tfm_data/test_train_n_shap_dilation/{site}/trained_models/member_{seed}_{name_save_CombinedModel}_{site}_test.pth")
    model.load_state_dict(model_state_dict)
    model.eval()


    # ---------------------------------------------------------------------------------
    
    background_indices = np.random.choice(len(train_subset_combined), 100, replace=False)
    background_nn = []
    background_cnn = []
    
    for idx in background_indices:
        nn_input, cnn_input, labels = train_subset_combined[idx]  # Adjust based on your dataset structure
        background_nn.append(nn_input)
        background_cnn.append(cnn_input)
    
    
    device = next(model.parameters()).device
    
    background_nn = torch.stack(background_nn,dim=0).to(device)
    background_cnn = torch.stack(background_cnn,dim=0).to(device)
    
    explainer_indices = np.arange(0,len(combined_test_dataset),1)
    explain_nn = []
    explain_cnn = []
    #
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
    
    
    raw_shap_dict = {
        'nn': shap_values_nn_raw,
        'cnn': shap_values_cnn_raw,
    }
    
    with open(f'/gpfs/scratch/bsc32/bsc167965/tfm_data/test_train_n_shap_dilation/{site}/SHAP/80p_SHAP_values_GradientExplainer_not_filtered_{seed}_{number_lags}lags.pkl', 'wb') as f:
        pickle.dump(raw_shap_dict, f)
    
    print(f"Finished training and SHAP value computing for site: {site}")
    
