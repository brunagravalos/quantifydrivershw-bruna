# IMPORT NEEDED PACKAGES
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import torch
import yaml
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import re
import pickle
import gc
import tqdm
import sys
import importlib.resources as pkg_resources

import hydra
from omegaconf import DictConfig, OmegaConf

# PATH FIX FOR PROJECT
cwd = os.getcwd()
script_dir = os.path.dirname(os.path.realpath(__file__))
project_src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_src_dir not in sys.path:
    sys.path.append(project_src_dir)

from quantifydrivers import machine_learning, data_files
from quantifydrivers.machine_learning import convnext_functions

# SETS DETERMINISM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device set to: {device} ***")

try:
    torch.use_deterministic_algorithms(True)
    print("Using deterministic algorithms.")
except Exception as e:
    print(f"Could not enforce deterministic algorithms: {e}")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    # Create torch generator seeded from config
    g = torch.Generator()
    g.manual_seed(cfg.SEED)

    # --- Build datasets
    from dataloading_script import build_datasets_and_loaders
    datasets = build_datasets_and_loaders(configuration=cfg, generator=g)

    # --- Run evaluation
    from evaluation_script import evaluation
    evaluation(configuration=cfg, datasets=datasets, generator=g, device=device)

    # --- SHAP computation
    from SHAP_script import compute_SHAP
    compute_SHAP(configuration=cfg, datasets=datasets, generator=g, device=device)


if __name__ == "__main__":
    main()
