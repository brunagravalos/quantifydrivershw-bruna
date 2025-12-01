# data_loading.py

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

import os, sys

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

from quantifydrivers import machine_learning

# -------------------------------------------------------
# Helper: load paths for (mock structure) dataset
# -------------------------------------------------------
def load_mock_paths(base_folder: str, site: str, percentile: str):
    return {
        "g500": os.path.join(base_folder, "mockLargeScale_data", "file_g500.nc"),
        "g200": os.path.join(base_folder, "mockLargeScale_data", "file_g200.nc"),
        "psl":  os.path.join(base_folder, "mockLargeScale_data", "file_psl.nc"),
        "co2":  os.path.join(base_folder, "mockLargeScale_data", "file_CO2.nc"),
        "local": os.path.join(base_folder, "mockLocalScale_data", f"file_local_{percentile}_{site}.nc")
    }


# -------------------------------------------------------
# Main entry point called by training script
# -------------------------------------------------------
def load_datasets_and_loaders(config_path: str):
    """
    Loads everything needed for training:
    - config YAML
    - datasets (local + large scale)
    - combined dataset
    - train/val/test dataloaders
    - metadata (hyperparameters, variables, epochs, paths)

    Returns:
      datasets: dict
      loaders: dict
      metadata: dict
    """

    # ------------------------------
    # 1. Read YAML config
    # ------------------------------
    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    SITE = CONF["SITE"]
    SEED = CONF["SEED"]
    percentile = CONF["percentile_to_load"]

    # ------------------------------
    # 2. Resolve file paths
    # ------------------------------
    paths = load_mock_paths(CONF["paths"]["base_folder"], SITE, percentile)

    file_local = paths["local"]
    file_g500  = paths["g500"]
    file_g200  = paths["g200"]
    file_psl   = paths["psl"]
    file_CO2   = paths["co2"]

    variables_era5     = CONF["dataset_config"]["variables_era5"]
    variables_era5land = CONF["dataset_config"]["variables_era5land"]
    start_date         = CONF["dataset_config"]["start_date"]

    # ------------------------------
    # 3. Build dataset configs
    # ------------------------------
    ERA5LAND_TRAIN = dict(
        start_date=start_date, end_date="2013-12-31",
        months=[6, 7, 8], variables=variables_era5land
    )

    ERA5LAND_TEST = dict(
        start_date="2014-01-01", end_date="2023-12-31",
        months=[6, 7, 8], variables=variables_era5land
    )

    ERA5_TRAIN = dict(
        start_date=start_date, end_date="2013-12-31",
        months=[6, 7, 8], start_lag=1, lags_era5=1, variables=variables_era5
    )

    ERA5_TEST = dict(
        start_date="2014-01-01", end_date="2023-12-31",
        months=[6, 7, 8], start_lag=1, lags_era5=1, variables=variables_era5
    )

    # ------------------------------
    # 4. Create datasets
    # ------------------------------
    train_local = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local, file_CO2=file_CO2, **ERA5LAND_TRAIN
    )

    test_local = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local, file_CO2=file_CO2, **ERA5LAND_TEST
    )

    train_large = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **ERA5_TRAIN
    )

    test_large = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **ERA5_TEST
    )

    # ------------------------------
    # 5. Combined dataset
    # ------------------------------
    combined_train = machine_learning.CombinedDataset(train_local, train_large, variables=variables_era5)
    combined_test  = machine_learning.CombinedDataset(test_local,  test_large,  variables=variables_era5)

    # ------------------------------
    # 6. Train/val split
    # ------------------------------
    g = torch.Generator().manual_seed(SEED)
    train_size = int(0.8 * len(combined_train))
    val_size   = len(combined_train) - train_size

    train_subset, val_subset = random_split(combined_train, [train_size, val_size], generator=g)

    # ------------------------------
    # 7. Create dataloaders
    # ------------------------------
    dl_conf = dict(
        batch_size=CONF["site_hypms"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=g
    )

    dl_test_conf = dict(
        batch_size=CONF["site_hypms"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    loaders = {
        "train": DataLoader(train_subset, **dl_conf),
        "val":   DataLoader(val_subset,   **dl_conf),
        "test":  DataLoader(combined_test, **dl_test_conf),
    }

    datasets = {
        "train_local": train_local,
        "test_local":  test_local,
        "train_large": train_large,
        "test_large":  test_large,
        "combined_train": combined_train,
        "combined_test": combined_test,
    }

    return datasets, loaders
