# IMPORT NEEDED PACKAGES ===============================================================================================
# ======================================================================================================================

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

# I HAD TO ADD THIS TO BE ABLE TO IMPORT MACHINE LEARNING MODULE =======================================================
# ======================================================================================================================
cwd = os.getcwd()
script_dir = os.path.dirname(os.path.realpath(__file__))
project_src_dir = os.path.abspath(os.path.join(script_dir, '..', '..')) # <--- **CHANGED TO TWO '..'**
if project_src_dir not in sys.path:
    sys.path.append(project_src_dir)
from quantifydrivers import machine_learning, data_files
from quantifydrivers.machine_learning import convnext_functions
# ======================================================================================================================
# ======================================================================================================================


# DETERMINISM ==========================================================================================================
# ======================================================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device set to: {device} ***") # NEW PRINT

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
# ======================================================================================================================

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "configuration.yaml")
with open(CONF_PATH, "r") as f:
    CONF = yaml.safe_load(f)
g = torch.Generator()

from data_loading2 import build_datasets_and_loaders
datasets = build_datasets_and_loaders(configuration=CONF,generator=g)

from training_pipeline import training
model, CNN_model_loaded, NN_model, losses_train_combined, losses_val_combined = training(configuration=CONF,datasets=datasets,device=device,generator=g)

from evaluation_pipeline import evaluation
evaluation(configuration=CONF,datasets=datasets,generator=g,losses_train_combined=losses_train_combined,losses_val_combined=losses_val_combined,model=model,CNN_model_loaded=CNN_model_loaded,NN_model=NN_model)

from SHAP_computing_pipeline import compute_SHAP
compute_SHAP(configuration=CONF,datasets=datasets,generator=g,device=device)