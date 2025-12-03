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


print("--- DIAGNOSTICS START ---")

# 1. Get and print the current working directory
cwd = os.getcwd()
print(f"1. Current Working Directory (os.getcwd()): {cwd}")

# 2. Calculate the project src directory
script_dir = os.path.dirname(os.path.realpath(__file__))
# Moves up 3 levels: train_and_shap -> quantifydrivers -> src
project_src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # <--- **CHANGED TO TWO '..'**
print(f"2. Calculated project_src_dir (expected): {project_src_dir}")

# 3. Add the path (if not already present)
if project_src_dir not in sys.path:
    sys.path.append(project_src_dir)

# 4. Print the final sys.path
print("\n3. Final sys.path content:")
for p in sys.path:
    print(f"- {p}")

from quantifydrivers import machine_learning, data_files

print("--- DIAGNOSTICS END ---")
from quantifydrivers.machine_learning import convnext_functions
# ======================================================================================================

# DEFINE DEVICE ----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** DEVICE: Model will run on {device} ***")
if device.type == 'cuda':
    print(f"*** CUDA Device Name: {torch.cuda.get_device_name(0)} ***")

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
print("*** Determinism settings confirmed: cudnn.benchmark=False, deterministic=True, TF32=False ***")


from data_loading import load_datasets_and_loaders

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
    print("--- DETERMINISM VERIFICATION ---")
    # Check PyTorch
    print(f"PyTorch rand(1): {torch.rand(1).item()}")
    # Check NumPy
    print(f"NumPy rand(): {np.random.rand()}")
    # Check Python random
    print(f"Python random(): {random.random()}")
    print("------------------------------")


def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The global 'g' generator is used later, we cannot access it here easily.
    # Assuming 'g' is handled correctly where it's defined (later in the script).
    print(f"*** SEEDS reset to {seed} ***")


def generate_ensemble_seeds(fixed_seed=123):
    print(f"*** Generating ensemble seeds using fixed seed: {fixed_seed} ***")
    rng = np.random.default_rng(fixed_seed)
    seeds = rng.integers(low=0, high=2 ** 32 - 1, size=20).tolist()
    print(f"*** Generated {len(seeds)} ensemble seeds. ***")
    return seeds


check_seeds()
# Generate list of seeds for the ensamble ---------------------------------------------------------------
list_seeds = generate_ensemble_seeds(fixed_seed=123)
# (local,regional,labels)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")
datasets, loaders = load_datasets_and_loaders(CONF_PATH)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")

with open(CONF_PATH, "r") as f:
    CONF = yaml.safe_load(f)
print("*** CONFIG: Configuration reloaded to capture hyperparameter updates. ***")

percentile_to_load = CONF["percentile_to_load"]
print(f"*** CONFIG: Target Percentile: {percentile_to_load} ***")

save_base = CONF["save_paths"]["model_dir"]
results_base = CONF["save_paths"]["results_dir"]

# ======================================================================================================
# DEFINE CONFIGURATION VALUES
# ======================================================================================================

# Name to save the trained CombinedModel
name_save_CombinedModel = (
    "CO2_Combinedmodel_trained_with_cnn_nn_trained_together_"
    f"{CONF['dataset_config']['variables_era5']}lags"
)
save_path = f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{CONF["SITE"]}/trained_models/member_{CONF["SEED"]}_{name_save_CombinedModel}_{CONF["SITE"]}_test_2.pth"
main_path = '/gpfs/scratch/bsc32/bsc214253/results/'  # Change to your desired path

# ======================================================================================================
# END OF CONFIG
# ======================================================================================================


# Get site from bash argument -------------------------------------------------------------------------
seed_to_process = int(CONF["SEED"])
print(f"Doing site: {CONF["SITE"]} with SEED: {seed_to_process}")

# =================================================================================================================
# Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
# ================================================================================================================

# =========================================================================================
# Dataloaders configuration dictionaries --------------------------------------------------
# =========================================================================================


print(f"*** TRAIN CONFIG: PyTorch generator 'g' initialized with seed {CONF["SEED"]} ***")

HYPMS = dict(
    epochs=CONF["epoch_config"]["epochs"],
    lr=CONF['site_hypms']['lr'],
    w_decay=CONF['site_hypms']['w_decay'],
)
print("*** HYPERPARAMETERS FOR TRAINING ***")
print(HYPMS)
print(f"Learning Rate Type Check: {type(CONF['site_hypms']['lr'])}")
print(f"Weight Decay: {HYPMS['w_decay']}")


#  Weights class imbalance  ---------------------------------------------------------------------------------

############################   UNTIL HERE WE ARA DATA LOADING        #########################################

unique_classes, class_counts = np.unique(datasets["train_local"].labels, return_counts=True)
total_counts = sum(class_counts)
print(f"*** CLASS WEIGHTS: Class counts (nonextreme, extreme): {class_counts} (Total: {total_counts}) ***")

# Alternative way to compute class weights, if wanted to use weights for each class -------------------------
# extreme_weights_ctt = SITE_HYPMS[site]['extreme_weights_ctt']
# nonextreme_weights_ctt = SITE_HYPMS[site]['nonextreme_weights_ctt']
# class_weights = torch.tensor([total_counts / (nonextreme_weights_ctt*class_counts[0]), total_counts / (extreme_weights_ctt*class_counts[1])], dtype=torch.float)
# -------------------------------------------------------------------------------------------------------------

base_minority_weight = class_counts[0] / class_counts[1]
print(f"*** CLASS WEIGHTS: Base minority weight (Non-extreme/Extreme): {base_minority_weight:.4f} ***")

minority_weight_multiplier = CONF['site_hypms']['minority_weight_multiplier']
final_minority_weight = base_minority_weight * minority_weight_multiplier
class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights

print(f"*** CLASS WEIGHTS: Multiplier: {minority_weight_multiplier} ***")
print(f"*** CLASS WEIGHTS: Final (Smoothed) Weights: {smoothed_weights.cpu().numpy()} (on {device}) ***")

# Cross-entropy loss criterion with class weights
criterion = nn.CrossEntropyLoss(weight=smoothed_weights)
print("*** CRITERION: nn.CrossEntropyLoss initialized with smoothed class weights. ***")

# Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------

reset_seeds(CONF["SEED"])
# MLP for local-scale
NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(datasets["train_local"].all_features),
                                                       train_alone_NN=False, num_classes=2).to(device)
print(f"*** MODEL INIT: NN Model initialized with input dim {len(datasets["train_local"].all_features)}. ***")

reset_seeds(CONF["SEED"])
# ConvNext for large-scale
CNN_model_loaded = convnext_functions.ConvNext(
    num_channels=len(datasets["train_large"].all_features),
    num_classes=2,
    patch_size=4,
    layer_dims=[4, 6, 6, 16],
    depths=[1, 2, 2, 1],
    drop_rate=0.05,
    train_alone=False,
).to(device)
print(f"*** MODEL INIT: CNN Model initialized with {len(datasets["train_large"].all_features)} input channels. ***")

reset_seeds(CONF["SEED"])

# Combined model
model = machine_learning.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                       output_dim=2).to(device)
print("*** MODEL INIT: CombinedModel initialized. ***")

reset_seeds(CONF["SEED"])

# =======================================================================================================================================
# Train phase Combined model -------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================

reset_seeds(CONF["SEED"])
# Optimizer
optimizer_combined = optim.AdamW(model.parameters(), lr=HYPMS['lr'], weight_decay=HYPMS['w_decay'])
print(f"*** OPTIMIZER: AdamW initialized with LR={HYPMS['lr']} and Weight Decay={HYPMS['w_decay']}. ***")

# Scheduler (if wanted)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_combined, T_max=30)
print("*** SCHEDULER: CosineAnnealingLR initialized with T_max=30. ***")


# Start training
print("\n====================================================================================")
print(f"|                  STARTING TRAINING OF COMBINED MODEL (Epochs: {HYPMS['epochs']})                |")
print("====================================================================================")
losses_train_combined, losses_val_combined, num_e, best_val_loss = machine_learning.train_CombinedModel(model,
                                                                                                        loaders["train"],
                                                                                                        loaders["val"],
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
print("====================================================================================")
print(f"|                  TRAINING COMPLETE. Ran for {num_e} epochs.                     |")
print("====================================================================================")

# Save the trained CombinedModel -----------------------------

save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    print(f"*** SAVE: Creating output directory: {save_dir} ***")
    os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(),save_path)
print(f"*** SAVE: CombinedModel state dictionary saved to: {save_path} ***")


# =======================================================================================================================================
# Evaluation phase ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================

print("\n====================================================================================")
print("|                        STARTING MODEL EVALUATION PHASE                           |")
print("====================================================================================")

reset_seeds(CONF["SEED"])
y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
    CombinedModel=model, cnn=CNN_model_loaded, nn=NN_model, test_loader=loaders["test"],
    print_accuracies=True, train_alone=False)
reset_seeds(CONF["SEED"])

print(f"*** EVALUATION RESULTS: Extreme Accuracy: {extreme_acc:.4f}, Non-extreme Accuracy: {nonextreme_acc:.4f} ***")

# Save the dictionaries with the relevant data ---------------------------------------------------------------------
seed_results = {
    'y_true_pred_pairs': (y_true, y_pred),  # These are from the current site
    'out_probs_seed': outputs_prob,
    'extreme_accuracy': extreme_acc,
    'nonextreme_accuracy': nonextreme_acc,
    'losses_train': losses_train_combined,
    'losses_val': losses_val_combined
}

results_file_path = os.path.join(main_path, f'{CONF["SITE"]}/{percentile_to_load}_results_data_{CONF["SEED"]}.pkl')

results_dir = os.path.dirname(results_file_path)
if not os.path.exists(results_dir):
    print(f"*** SAVE: Creating results directory: {results_dir} ***")
    os.makedirs(results_dir, exist_ok=True)


with open(results_file_path, 'wb') as f:
    pickle.dump(seed_results, f)
    print(f"*** SAVE: Results dictionary saved to: {results_file_path} ***")

print("Finished training model, computing SHAP")

# =======================================================================================================================================
# SHAP computation ----------------------------------------------------------------------------------------------------------------------
# =======================================================================================================================================

print("\n====================================================================================")
print("|                          STARTING SHAP COMPUTATION                               |")
print("====================================================================================")

# Prepare NN model and CNN model for SHAP -----------------------------------------------------------------------------------------------
NN_model_loaded = machine_learning.ToCombineExtremeClassifier(input_dim=len(datasets["train_local"].all_features),
                                                              train_alone_NN=False, num_classes=2).to(device)
NN_model_loaded.eval()
print("*** SHAP Prep: NN Model loaded and set to eval mode. ***")
reset_seeds(CONF["SEED"])

CNN_model_loaded = convnext_functions.ConvNext(
    num_channels=len(datasets["train_large"].all_features),
    num_classes=2,
    patch_size=4,
    layer_dims=[4, 6, 6, 16],
    depths=[1, 2, 2, 1],
    drop_rate=0.05,
    train_alone=False,
).to(device)
print("*** SHAP Prep: CNN Model loaded. ***")
reset_seeds(CONF["SEED"])

# Create the Combined model for SHAP---------------------------------------------------------------------------------------------------
model = machine_learning.CombinedModel(NN_model_loaded, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                       output_dim=2).to(device)
print("*** SHAP Prep: Combined Model re-initialized for SHAP. ***")
reset_seeds(CONF["SEED"])

# Load the CONF["SEED"] CombinedModel weights ----------------------------------------------------------------------------------------------
print(f"*** SHAP Prep: Loading weights from {save_path}... ***")
model_state_dict = torch.load(
    save_path,weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()
print("*** SHAP Prep: Weights loaded and model set to eval mode. ***")

# ---------------------------------------------------------------------------------------------------------------------

# Create baseline for SHAP computation --------------------------------------------------------------------------------
background_indices = np.random.choice(len(datasets["train_subset"]), 200, replace=False)
background_nn = []
background_cnn = []

print(f"*** SHAP Data: Creating background dataset (200 samples from train_subset). ***")
for idx in tqdm.tqdm(background_indices, desc="Preparing SHAP Background"):
    nn_input, cnn_input, labels = datasets["train_subset"][idx]  # Adjust based on your dataset structure
    background_nn.append(nn_input)
    background_cnn.append(cnn_input)

device = next(model.parameters()).device

background_nn = torch.stack(background_nn, dim=0).to(device)
background_cnn = torch.stack(background_cnn, dim=0).to(device)
print(f"*** SHAP Data: Background NN shape: {background_nn.shape}, CNN shape: {background_cnn.shape} ***")

# Create explainer dataset for SHAP computation -----------------------------------------------------------------------
explainer_indices = np.arange(0, len(datasets["combined_test"]), 1)
explain_nn = []
explain_cnn = []

print(f"*** SHAP Data: Creating explanation dataset ({len(explainer_indices)} samples from combined_test). ***")
for idx in tqdm.tqdm(explainer_indices, desc="Preparing SHAP Explain Data"):
    nn_input, cnn_input, labels = datasets["combined_test"][idx]
    explain_nn.append(nn_input)
    explain_cnn.append(cnn_input)

explain_nn = torch.stack(explain_nn, dim=0).to(device)
explain_cnn = torch.stack(explain_cnn, dim=0).to(device)
print(f"*** SHAP Data: Explain NN shape: {explain_nn.shape}, CNN shape: {explain_cnn.shape} ***")

# Merge local-scale and large-scale data for SHAP computation ------------------------------------------------------
background_data = [background_nn, background_cnn]
explain_data = [explain_nn, explain_cnn]

reset_seeds(CONF["SEED"])
print("Initializing GradientExplainer...")
explainer_grad = shap.GradientExplainer(model, background_data)
print("Explainer initialized.")
reset_seeds(CONF["SEED"])
print("Calculating SHAP values...")
shap_values = explainer_grad.shap_values(explain_data)

print(f"*** SHAP Computation complete. ***")
print(f"Finished computing SHAP values for SITE: {CONF["SITE"]}")

# Select class to explaine, extreme (1) in our case ----------------------------------------------------------------
class_index_to_explain = 1
shap_values_nn_raw = shap_values[0][:, :, 1]  # NumPy array (N_explain, nn_features)
shap_values_cnn_raw = shap_values[1][:, :, :, :, 1]  # NumPy array (N_explain, V, H, W)
print(f"*** SHAP Results: Extracted NN SHAP values for class 1, shape: {shap_values_nn_raw.shape} ***")
print(f"*** SHAP Results: Extracted CNN SHAP values for class 1, shape: {shap_values_cnn_raw.shape} ***")


# Dictionary to save SHAP values -----------------------------------------------------------------------------------
raw_shap_dict = {
    'nn': shap_values_nn_raw,
    'cnn': shap_values_cnn_raw,
}

# with open(f'/your/path/to/save/SHAP/results', 'wb') as f:
#   pickle.dump(raw_shap_dict, f)

print(f"Finished training and SHAP value computing for site: {CONF["SITE"]}")
print("====================================================================================")