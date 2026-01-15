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
from datetime import datetime

# PATH FIX FOR PROJECT
cwd = os.getcwd()
script_dir = os.path.dirname(os.path.realpath(__file__))
project_src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_src_dir not in sys.path:
    sys.path.append(project_src_dir)


from quantifydrivers.train_and_shap.config_schema import validate_schema

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


def save_used_config(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "used_config.yaml")
    OmegaConf.save(cfg, config_path)
    print(f"*** Saved used configuration to {config_path} ***")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    timestamp = datetime.now()
    formatted_time = timestamp.strftime('%m-%d-%Y_%H-%M')
    print(formatted_time)
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    try:
        validated_cfg = validate_schema(cfg)
    except Exception as e:
        print("\nCONFIG VALIDATION FAILED")
        print(e)
        raise

    print("\nConfig validation passed!")
    print(validated_cfg)

    # Create torch generator seeded from config
    g = torch.Generator()
    g.manual_seed(validated_cfg.seed)

    # --- Build datasets
    from dataloading_script import build_datasets_and_loaders
    datasets = build_datasets_and_loaders(configuration=validated_cfg, generator=g)

    # --- Train model
    from training_script import training
    training(configuration=validated_cfg,datasets=datasets,device=device,generator=g,timestamp=formatted_time)

    # --- Run evaluation
    from evaluation_script import evaluation
    evaluation(configuration=validated_cfg, datasets=datasets, generator=g, device=device, timestamp=formatted_time)

    # --- SHAP computation
    from SHAP_script import compute_SHAP
    compute_SHAP(configuration=validated_cfg, datasets=datasets, generator=g, device=device, timestamp=formatted_time)

    # --- Saved used configuration
    results_dir = os.path.join(
        validated_cfg.paths.results_dir,
        validated_cfg.site.name,
        f"{validated_cfg.site.name}_{validated_cfg.percentile}_{validated_cfg.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)
    save_used_config(cfg, results_dir)


if __name__ == "__main__":
    main()
