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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import re
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


# DEFINE DEVICE --------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device set to: {device} ***") # NEW PRINT

# CECK DETERMINISM -----------------------------------------------------------------------------------------------------

try:
    torch.use_deterministic_algorithms(True)
    print("Using deterministic algorithms.")
except Exception as e:
    print(f"Could not enforce deterministic algorithms: {e}")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# ======================================================================================================================
# DEFINE FUNCTIONS: seed treatment, loading of hyperparameters from hypm optimization
# ======================================================================================================================

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

check_seeds()

# Generate list of seeds for the ensamble ---------------------------------------------------------------
list_seeds = generate_ensemble_seeds(fixed_seed=123)

# Get site from bash argument -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train combined model for a specific seed.")
parser.add_argument("seed_value", type=str, help="seed value, member of ensamble")
args = parser.parse_args()
seed_to_process = int(args.seed_value)
print(f"*** Ensemble seeds generated. Using seed: {seed_to_process} ***") # NEW PRINT

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")

with open(CONF_PATH, "r") as f:
    CONF = yaml.safe_load(f)

seed = CONF["SEED"]
SITE = 'cordoba'
print(f"Doing site: {SITE} with SEED: {seed_to_process}")

g = torch.Generator()

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")

from data_loading2 import build_datasets_and_loaders
datasets = build_datasets_and_loaders(CONF=CONF,seed=seed,generator=g)

from training_pipeline import training
model, CNN_model_loaded, NN_model, losses_train_combined, losses_val_combined = training(CONF_PATH,datasets,seed,device,g)

from evaluation_pipeline import evaluation
evaluation(CONF_PATH,datasets, seed, g, losses_train_combined, losses_val_combined, model,CNN_model_loaded, NN_model)

from SHAP_computing_pipeline import compute_SHAP
compute_SHAP(CONF_PATH,datasets, seed, g, device)