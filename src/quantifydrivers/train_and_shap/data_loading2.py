# data_loading.py
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import yaml
import sys
import os

from quantifydrivers import machine_learning

def load_mock_paths(base_folder: str, site: str, percentile: str):
    return {
        "g500": os.path.join(base_folder, "mockLargeScale_data", "file_g500.nc"),
        "g200": os.path.join(base_folder, "mockLargeScale_data", "file_g200.nc"),
        "psl": os.path.join(base_folder, "mockLargeScale_data", "file_psl.nc"),
        "co2": os.path.join(base_folder, "mockLargeScale_data", "file_CO2.nc"),
        "local": os.path.join(base_folder, "mockLocalScale_data", f"file_local_{percentile}_{site}.nc")
    }

# ======================================================================
# FUNCTION TO CREATE ALL DATASETS & DATALOADERS
# ======================================================================


def build_datasets_and_loaders(
        config_path,
        dataloader_conf,
        dataloader_test_conf,
        seed):

    #### OPEN CONFIG ####

    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    SITE = CONF["SITE"]
    SEED = CONF["SEED"]

    percentile = CONF["percentile_to_load"]

    paths = load_mock_paths(CONF["paths"]["base_folder"], SITE, percentile)

    file_local_scale = paths["local"]
    file_g500 = paths["g500"]
    file_g200 = paths["g200"]
    file_psl = paths["psl"]
    file_CO2 = paths["co2"]
    variables_era5 = CONF["dataset_config"]["variables_era5"]
    variables_era5land = CONF["dataset_config"]["variables_era5land"]
    start_date = CONF["dataset_config"]["start_date"]

    era5land_train_conf = dict(
        start_date=start_date, end_date="2013-12-31",
        months=[6, 7, 8], variables=variables_era5land
    )

    era5land_test_conf = dict(
        start_date="2014-01-01", end_date="2023-12-31",
        months=[6, 7, 8], variables=variables_era5land
    )

    era5_train_conf = dict(
        start_date=start_date, end_date="2013-12-31",
        months=[6, 7, 8], start_lag=1, lags_era5=1, variables=variables_era5
    )

    era5_test_conf = dict(
        start_date="2014-01-01", end_date="2023-12-31",
        months=[6, 7, 8], start_lag=1, lags_era5=1, variables=variables_era5
    )

    # -------------------------------
    # 1. CREATE LOCAL-SCALE DATASETS
    # -------------------------------
    train_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local_scale, file_CO2=file_CO2, **era5land_train_conf)

    test_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local_scale, file_CO2=file_CO2, **era5land_test_conf)

    # --------------------------------
    # 2. CREATE LARGE-SCALE DATASETS
    # --------------------------------
    train_features_era5 = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **era5_train_conf)

    test_features_era5 = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **era5_test_conf)

    # ---------------------------------------
    # 3. COMBINED DATASET
    # ---------------------------------------
    combined_train = machine_learning.CombinedDataset(
        train_dataset, train_features_era5, variables=variables_era5)

    combined_test = machine_learning.CombinedDataset(
        test_dataset, test_features_era5, variables=variables_era5)

    # ---------------------------------------
    # 4. TRAIN/VAL SPLIT
    # ---------------------------------------
    g = torch.Generator().manual_seed(seed)

    train_size = int(0.8 * len(combined_train))
    val_size = len(combined_train) - train_size

    train_subset, val_subset = random_split(
        combined_train, [train_size, val_size], generator=g)

    # ---------------------------------------
    # 5. DATALOADERS
    # ---------------------------------------
    train_loader = DataLoader(train_subset, **dataloader_conf)
    val_loader   = DataLoader(val_subset,   **dataloader_conf)
    test_loader  = DataLoader(combined_test, **dataloader_test_conf)


    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_era5": train_features_era5,
        "test_era5": test_features_era5,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_subset": train_subset,
        "test_subset": val_subset,
        "combined_train": combined_train,
        "combined_test": combined_test,
    }
