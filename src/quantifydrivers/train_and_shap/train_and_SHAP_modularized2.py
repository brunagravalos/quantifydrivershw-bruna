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
print(f"*** Device set to: {device} ***") # NEW PRINT

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
    print("--- CURRENT SEED STATES ---")
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
    print(f"*** Generating ensemble seeds using fixed seed: {fixed_seed} ***")
    rng = np.random.default_rng(fixed_seed)
    seeds = rng.integers(low=0, high=2 ** 32 - 1, size=20).tolist()
    print(f"*** Generated {len(seeds)} ensemble seeds. ***")
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
    print(f"*** Checking hyperparameter file path: {file_path} ***") # NEW PRINT

    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None
    print(f"*** Hyperparameter file found for {site_name}. Loading content... ***") # NEW PRINT

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
    print(f"*** Hyperparameters successfully parsed. ***") # NEW PRINT
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
print(f"*** Ensemble seeds generated. Using seed: {seed_to_process} ***") # NEW PRINT


# Get seed to process from bash script
seed = seed_to_process


# Define sites and base hyperparameters ----------------------------------------------------------------------

SITE = 'cordoba'
print(f"Doing site: {SITE} with SEED: {seed_to_process}")


SITE_HYPMS_fixed = {
    'belgrado': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
    'hannover': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
    'stockholm': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
    'lyon': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
    'cordoba': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
    'marrakech': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1}}

# Create empty dictionary with the base HYPMS
SITE_HYPMS = SITE_HYPMS_fixed.copy()

# ==============================================================================================================
# Choose which percentile's hyperparameters to load
percentile_to_load = '90p'

print(f"Loading hyperparameters for percentile: {percentile_to_load}")
params = load_hypms_from_file(SITE, percentile=percentile_to_load, file_name="file_with_hypms.txt")
if params:
    SITE_HYPMS[SITE] = params
print(f"Loaded hyperparameters for {SITE}: {SITE_HYPMS[SITE]}")


print(f"Doing site: {SITE}")
print(f"*** Setting up file paths for site: {SITE} ***") # NEW PRINT


# =================================================================================================================
# Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
# =================================================================================================================

HYPMS = dict(
    epochs=75,
    lr=SITE_HYPMS[SITE]['lr'],
    w_decay=SITE_HYPMS[SITE]['w_decay'],
)

g = torch.Generator()
reset_seeds(seed)


batch_size = 32

from dataloading_script import build_datasets_and_loaders

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")

datasets = build_datasets_and_loaders(
    config_path=CONF_PATH,
    seed=seed,generator=g)

train_dataset = datasets["train_dataset"]
test_dataset = datasets["test_dataset"]
train_features_era5 = datasets["train_era5"]
test_features_era5 = datasets["test_era5"]
train_subset_combined = datasets["train_subset"]

combined_train_loader = datasets["train_loader"]
combined_val_loader   = datasets["val_loader"]
combined_test_loader  = datasets["test_loader"]
combined_test_dataset = datasets["combined_test"]

number_lags = ['g500', 'g200', 'psl']

# Name to save the trained CombinedModel
name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"


#  Weights class imbalance  ---------------------------------------------------------------------------------
print("*** Calculating Class Weights ***") # NEW PRINT

unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
print(f"*** Class counts (0: non-extreme, 1: extreme): {class_counts} ***") # NEW PRINT
total_counts = sum(class_counts)

base_minority_weight = class_counts[0] / class_counts[1]

minority_weight_multiplier = SITE_HYPMS[SITE]['minority_weight_multiplier']
final_minority_weight = base_minority_weight * minority_weight_multiplier
class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights
print(f"*** Smoothed Class Weights (0, 1): {smoothed_weights.cpu().numpy()} ***") # NEW PRINT

# Cross-entropy loss criterion with class weights
criterion = nn.CrossEntropyLoss(weight=smoothed_weights)

# Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
print("*** Initializing Models (NN and CNN) ***") # NEW PRINT

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
print("*** Combined Model initialized. Starting Training Phase... ***") # NEW PRINT

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
save_path = f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{seed}_{name_save_CombinedModel}_{SITE}_test_2.pth"

save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)  # os.makedirs creates all intermediate folders
    print(f"Created output directory: {save_dir}")  # Optional: Confirmation print
torch.save(model.state_dict(),
    f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{seed}_{name_save_CombinedModel}_{SITE}_test_2.pth")
print(f"*** Trained model state dictionary saved to: {save_path} ***") # NEW PRINT

# =======================================================================================================================================
# Evaluation phase ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================
print("*** Starting Evaluation Phase on Test Data ***") # NEW PRINT

reset_seeds(seed)
y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
    CombinedModel=model, cnn=CNN_model_loaded, nn=NN_model, test_loader=combined_test_loader,
    print_accuracies=True, train_alone=False)
reset_seeds(seed)
print(f"*** Evaluation complete. Extreme Accuracy: {extreme_acc:.4f}, Non-Extreme Accuracy: {nonextreme_acc:.4f} ***") # NEW PRINT

# Save the dictionaries with the relevant data ---------------------------------------------------------------------
seed_results = {
    'y_true_pred_pairs': (y_true, y_pred),  # These are from the current site
    'out_probs_seed': outputs_prob,
    'extreme_accuracy': extreme_acc,
    'nonextreme_accuracy': nonextreme_acc,
    'losses_train': losses_train_combined,
    'losses_val': losses_val_combined
}

main_path = '/gpfs/scratch/bsc32/bsc214253/results/'  # Change to your desired path
results_file_path = os.path.join(main_path, f'{SITE}/{percentile_to_load}_results_data_{seed}.pkl')

results_dir = os.path.dirname(results_file_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)


with open(results_file_path, 'wb') as f:
    pickle.dump(seed_results, f)
    print(f"saved file results {seed}") # ORIGINAL PRINT
    print(f"*** Results dictionary saved to: {results_file_path} ***") # NEW PRINT

print("Finished training model, computing SHAP")

# =======================================================================================================================================
# SHAP computation ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================
print("*** Starting SHAP Computation Phase ***") # NEW PRINT

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
print(f"*** Loading trained weights from: {save_path} for SHAP ***") # NEW PRINT
model_state_dict = torch.load(
    f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{seed}_{name_save_CombinedModel}_{SITE}_test_2.pth",weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()

# ---------------------------------------------------------------------------------------------------------------------

# Create baseline for SHAP computation --------------------------------------------------------------------------------
print(f"*** Creating SHAP background dataset (200 samples) ***") # NEW PRINT
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
print(f"*** Creating SHAP explanation dataset ({len(combined_test_dataset)} samples) ***") # NEW PRINT
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

print(f"Finished computing SHAP values for SITE: {SITE}") # CHANGED 'site' to 'SITE'

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

print(f"Finished training and SHAP value computing for site: {SITE}") # CHANGED 'site' to 'SITE'