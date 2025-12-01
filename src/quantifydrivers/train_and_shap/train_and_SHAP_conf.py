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
import sys
import os
import importlib.resources as pkg_resources
import yaml

from data_loading import load_datasets_and_loaders


print("--- DIAGNOSTICS START ---")

# 1. Get and print the current working directory
cwd = os.getcwd()
print(f"1. Current Working Directory (os.getcwd()): {cwd}")

# 2. Calculate the project src directory
script_dir = os.path.dirname(os.path.realpath(__file__))
# Moves up 3 levels: train_and_shap -> quantifydrivers -> src
project_src_dir = os.path.abspath(os.path.join(script_dir, '..', '..')) # <--- **CHANGED TO TWO '..'**
print(f"2. Calculated project_src_dir (expected): {project_src_dir}")

# 3. Add the path (if not already present)
if project_src_dir not in sys.path:
    sys.path.append(project_src_dir)

# 4. Print the final sys.path
print("\n3. Final sys.path content:")
for p in sys.path:
    print(f"- {p}")

from quantifydrivers import machine_learning, data_files
from quantifydrivers.machine_learning import convnext_functions
print("--- DIAGNOSTICS END ---")
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
    print(f"PyTorch rand(): {torch.rand(1).item()}")
    # Check NumPy
    print(f"NumPy rand(): {np.random.rand()}")
    # Check Python random
    print(f"Python random(): {random.random()}")


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)


def generate_ensemble_seeds(fixed_seed=123):
    rng = np.random.default_rng(fixed_seed)
    seeds = rng.integers(low=0, high=2 ** 32 - 1, size=20).tolist()
    return seeds


def load_hypms_from_file(site_name, percentile='90p', base_path='/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs',
                         file_name=None):
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
    #file_path = os.path.join(base_path, file_name)

    #file_path = pkg_resources.files(data_files).joinpath("HYPMS_optimization_results").joinpath(f"1lag_{site_name}_best_params_{percentile}_with_testing_phase.txt")

    # 1. Get the directory of the current script:
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. Go up two levels to reach the 'src' directory, then navigate down into the data files.
    # The path needs to be: /quantifydrivershw/src/quantifydrivers/data_files/HYPMS_optimization_results/

    # Path to 'quantifydrivers' directory
    quantifydrivers_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Construct the final path using os.path.join for reliability
    file_path = os.path.join(
        quantifydrivers_dir,
        "data_files",
        "HYPMS_optimization_results",
        f"g500_1lag_{site_name}_best_params_{percentile}_with_testing_phase.txt"
    )

    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None

    # This mapping handles differences between keys in the file and keys in script code
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

import os

def load_mock_paths(mock_base: str, site: str, percentile: str):
    """
    Load paths from the mock data structure that mimics the final dataset layout.

    Parameters
    ----------
    mock_base : str
        Path to the mockdata/ folder that contains `mockLargeScale_data` and `mockLocalScale_data`.
    site : str
        Location name, lowercase (e.g., "cordoba").
    percentile : str
        Percentile code, e.g. "90p".

    Returns
    -------
    dict
        Dictionary with file paths.
    """

    # Large-scale mock files
    file_g500 = os.path.join(mock_base, "mockLargeScale_data", "file_g500.nc")
    file_g200 = os.path.join(mock_base, "mockLargeScale_data", "file_g200.nc")
    file_psl  = os.path.join(mock_base, "mockLargeScale_data", "file_psl.nc")
    file_co2 = os.path.join(mock_base, "mockLargeScale_data", "file_CO2.nc")

    # Local-scale mock file
    file_local = os.path.join(
        mock_base,
        "mockLocalScale_data",
        f"file_local_{percentile}_{site}.nc"
    )


    return {
        "g500": file_g500,
        "g200": file_g200,
        "psl": file_psl,
        "co2": file_co2,
        "local": file_local
        }

# -----------------------------
# Load configuration from YAML
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")

with open(CONF_PATH, "r") as f:
    CONF = yaml.safe_load(f)

SITE = CONF["SITE"]
SEED = CONF["SEED"]
percentile_to_load = CONF["percentile_to_load"]

paths = load_mock_paths(CONF["paths"]["base_folder"], site=SITE, percentile=percentile_to_load)

file_g500 = paths["g500"]
file_g200 = paths["g200"]
file_psl = paths["psl"]
file_CO2 = paths["co2"]
file_local_scale = paths["local"]

variables_era5 = CONF["dataset_config"]["variables_era5"]
variables_era5land = CONF["dataset_config"]["variables_era5land"]
start_date = CONF["dataset_config"]["start_date"]

EPOCHS = CONF["epoch_config"]["epochs"]

SITE_HYPMS_fixed = CONF["site_hypms"]
print(f"SITE_HYPMS_fixed: {SITE_HYPMS_fixed}")

save_base = CONF["save_paths"]["model_dir"]
results_base = CONF["save_paths"]["results_dir"]

# ======================================================================================================
# DEFINE CONFIGURATION VALUES
# ======================================================================================================


# WE SHOULD PROVIDE A MINORITY WEIGHT HERE ALSO
SITE_HYPMS_fixed = {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1}
print(f"SITE_HYPMS_fixed: {SITE_HYPMS_fixed}")

start_date = "1950-01-01"

# Local-scale datasets configuration
_ERA5LAND_TRAIN_DATASET_CONF = dict(
    start_date=start_date,
    end_date="2013-12-31",
    months=[6, 7, 8],
    variables=variables_era5land
)

_ERA5LAND_TEST_DATASET_CONF = dict(
    start_date="2014-01-01",
    end_date="2023-12-31",
    months=[6, 7, 8],
    variables=variables_era5land
)

# Large-scale datasets configuration
_ERA5_TRAIN_DATASET_CONF = dict(
    start_date=start_date,
    end_date="2013-12-31",
    months=[6, 7, 8],
    start_lag=1,
    lags_era5=1,
    variables=variables_era5
)

_ERA5_TEST_DATASET_CONF = dict(
    start_date="2014-01-01",
    end_date="2023-12-31",
    months=[6, 7, 8],
    start_lag=1,
    lags_era5=1,
    variables=variables_era5
)


# Number of lags large-scale fields
number_lags = _ERA5_TEST_DATASET_CONF['variables']

# Name to save the trained CombinedModel
name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
save_path = f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{SEED}_{name_save_CombinedModel}_{SITE}_test_2.pth"
main_path = '/gpfs/scratch/bsc32/bsc214253/results/'  # Change to your desired path


# ======================================================================================================
# END OF CONFIG
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
seed = SEED

# Create empty dictionary with the base HYPMS
SITE_HYPMS = SITE_HYPMS_fixed.copy()

# ==============================================================================================================
# Choose which percentile's hyperparameters to load

print(f"Loading hyperparameters for percentile: {percentile_to_load}")

# which script should this go to?
params = load_hypms_from_file(SITE, percentile=percentile_to_load, file_name="file_with_hypms.txt")

if params:
    SITE_HYPMS = params
print(f"Loaded hyperparameters for {SITE}: {SITE_HYPMS}")


print(f"Doing site: {SITE}")

# =================================================================================================================
# Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
# =================================================================================================================

HYPMS = dict(
    epochs=EPOCHS,
    lr=SITE_HYPMS['lr'],
    w_decay=SITE_HYPMS['w_decay'],
)

# Datasets ERA5land data --------------------------------------------------------------

train_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
    file_path=file_local_scale, file_CO2=file_CO2, **_ERA5LAND_TRAIN_DATASET_CONF)
test_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
    file_path=file_local_scale, file_CO2=file_CO2, **_ERA5LAND_TEST_DATASET_CONF)

# Datasets ERA5 data ------------------------------------------------------------------

train_features_era5 = machine_learning.LargeScale_Dataset_extremes(file_g500, file_g200, file_psl,
                                                                   **_ERA5_TRAIN_DATASET_CONF)  # shape: features, time, lat, lon
test_features_era5 = machine_learning.LargeScale_Dataset_extremes(file_g500, file_g200, file_psl,
                                                                  **_ERA5_TEST_DATASET_CONF)

# =========================================================================================
# Dataloaders configuration dictionaries --------------------------------------------------
# =========================================================================================

g = torch.Generator()
reset_seeds(seed)

_DATALOADERS_CONF = dict(
    batch_size=SITE_HYPMS['batch_size'],
    drop_last=False,
    shuffle=True,
    num_workers=0,
    generator=g
)

_DATALOADERS_TEST_CONF = dict(
    batch_size=SITE_HYPMS['batch_size'],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# =========================================================================================

# Combined Dataset and Dataloader

batch_size = _DATALOADERS_CONF['batch_size']  # batch size for dataloaders both datasets

combined_train_dataset = machine_learning.CombinedDataset(train_dataset, train_features_era5,
                                                          variables=variables_era5)
combined_test_dataset = machine_learning.CombinedDataset(test_dataset, test_features_era5, variables=variables_era5)

# Split train and validation sets for the combined dataset ------------------------------------------------
train_size_combined = int(0.8 * len(combined_train_dataset))
val_size_combined = len(combined_train_dataset) - train_size_combined

# random split
train_subset_combined, val_subset_combined = random_split(combined_train_dataset,
                                                          [train_size_combined, val_size_combined], generator=g)

# (local,regional,labels)

combined_train_loader = DataLoader(train_subset_combined, **_DATALOADERS_CONF)
combined_val_loader = DataLoader(val_subset_combined, **_DATALOADERS_CONF)
combined_test_loader = DataLoader(combined_test_dataset, **_DATALOADERS_TEST_CONF)

#  Weights class imbalance  ---------------------------------------------------------------------------------

###############################################      UNTIL HERE WE ARA DATA LOADING#################################################

unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
total_counts = sum(class_counts)

# Alternative way to compute class weights, if wanted to use weights for each class -------------------------
# extreme_weights_ctt = SITE_HYPMS[site]['extreme_weights_ctt']
# nonextreme_weights_ctt = SITE_HYPMS[site]['nonextreme_weights_ctt']
# class_weights = torch.tensor([total_counts / (nonextreme_weights_ctt*class_counts[0]), total_counts / (extreme_weights_ctt*class_counts[1])], dtype=torch.float)
# -------------------------------------------------------------------------------------------------------------

base_minority_weight = class_counts[0] / class_counts[1]

minority_weight_multiplier = SITE_HYPMS['minority_weight_multiplier']
final_minority_weight = base_minority_weight * minority_weight_multiplier
class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights

# Cross-entropy loss criterion with class weights
criterion = nn.CrossEntropyLoss(weight=smoothed_weights)

# Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------

reset_seeds(seed)
# MLP for local-scale
NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                       train_alone_NN=False, num_classes=2).to(device)
reset_seeds(seed)
# ConvNext for large-scale
CNN_model_loaded = convnext_functions.ConvNext(
    num_channels=len(train_features_era5.all_features),
    num_classes=2,
    patch_size=4,
    layer_dims=[4, 6, 6, 16],
    depths=[1, 2, 2, 1],
    drop_rate=0.05,
    train_alone=False,
).to(device)
reset_seeds(seed)

# Combined model
model = machine_learning.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                       output_dim=2).to(device)
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
print(" Training combined model ")
losses_train_combined, losses_val_combined, num_e, best_val_loss = machine_learning.train_CombinedModel(model,
                                                                                                        combined_train_loader,
                                                                                                        combined_val_loader,
                                                                                                        criterion=criterion,
                                                                                                        optimizer=optimizer_combined,
                                                                                                        num_epochs=
                                                                                                        HYPMS[
                                                                                                            'epochs'],
                                                                                                        plot_loss=False,
                                                                                                        print_loss=False,
                                                                                                        early_stop=True,
                                                                                                        patience=5,
                                                                                                        print_early_stop=False,
                                                                                                        trial=None)

# Save the trained CombinedModel -----------------------------

save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)  # os.makedirs creates all intermediate folders
    print(f"Created output directory: {save_dir}")  # Optional: Confirmation print
torch.save(model.state_dict(),save_path)

# =======================================================================================================================================
# Evaluation phase ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================

reset_seeds(seed)
y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
    CombinedModel=model, cnn=CNN_model_loaded, nn=NN_model, test_loader=combined_test_loader,
    print_accuracies=True, train_alone=False)
reset_seeds(seed)

# Save the dictionaries with the relevant data ---------------------------------------------------------------------
seed_results = {
    'y_true_pred_pairs': (y_true, y_pred),  # These are from the current site
    'out_probs_seed': outputs_prob,
    'extreme_accuracy': extreme_acc,
    'nonextreme_accuracy': nonextreme_acc,
    'losses_train': losses_train_combined,
    'losses_val': losses_val_combined
}

results_file_path = os.path.join(main_path, f'{SITE}/{percentile_to_load}_results_data_{seed}.pkl')

results_dir = os.path.dirname(results_file_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)


with open(results_file_path, 'wb') as f:
    pickle.dump(seed_results, f)
    print(f"saved file results {seed}")

print("Finished training model, computing SHAP")

# =======================================================================================================================================
# SHAP computation ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================

# Prepare NN model and CNN model for SHAP -----------------------------------------------------------------------------------------------
NN_model_loaded = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                              train_alone_NN=False, num_classes=2).to(device)
NN_model_loaded.eval()
reset_seeds(seed)
CNN_model_loaded = convnext_functions.ConvNext(
    num_channels=len(train_features_era5.all_features),
    num_classes=2,
    patch_size=4,
    layer_dims=[4, 6, 6, 16],
    depths=[1, 2, 2, 1],
    drop_rate=0.05,
    train_alone=False,
).to(device)
reset_seeds(seed)

# Create the Combined model for SHAP---------------------------------------------------------------------------------------------------
model = machine_learning.CombinedModel(NN_model_loaded, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                       output_dim=2).to(device)
reset_seeds(seed)
# Load the trained CombinedModel weights ----------------------------------------------------------------------------------------------
model_state_dict = torch.load(
    save_path,weights_only=True)
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

background_nn = torch.stack(background_nn, dim=0).to(device)
background_cnn = torch.stack(background_cnn, dim=0).to(device)

# Create explainer dataset for SHAP computation -----------------------------------------------------------------------
explainer_indices = np.arange(0, len(combined_test_dataset), 1)
explain_nn = []
explain_cnn = []

for idx in explainer_indices:
    nn_input, cnn_input, labels = combined_test_dataset[idx]
    explain_nn.append(nn_input)
    explain_cnn.append(cnn_input)

explain_nn = torch.stack(explain_nn, dim=0).to(device)
explain_cnn = torch.stack(explain_cnn, dim=0).to(device)

# Merge local-scale and large-scale data for SHAP computation ------------------------------------------------------
background_data = [background_nn, background_cnn]
explain_data = [explain_nn, explain_cnn]

reset_seeds(seed)
print("Initializing GradientExplainer...")
explainer_grad = shap.GradientExplainer(model, background_data)
print("Explainer initialized.")
reset_seeds(seed)
print("Calculating SHAP values...")
shap_values = explainer_grad.shap_values(explain_data)

print(f"Finished computing SHAP values for SITE: {SITE}")

# Select class to explaine, extreme (1) in our case ----------------------------------------------------------------
class_index_to_explain = 1
shap_values_nn_raw = shap_values[0][:, :, 1]  # NumPy array (N_explain, nn_features)
shap_values_cnn_raw = shap_values[1][:, :, :, :, 1]  # NumPy array (N_explain, V, H, W)

# Dictionary to save SHAP values -----------------------------------------------------------------------------------
raw_shap_dict = {
    'nn': shap_values_nn_raw,
    'cnn': shap_values_cnn_raw,
}

# with open(f'/your/path/to/save/SHAP/results', 'wb') as f:
#   pickle.dump(raw_shap_dict, f)

print(f"Finished training and SHAP value computing for site: {SITE}")

