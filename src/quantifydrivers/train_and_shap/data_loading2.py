# data_loading.py
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import yaml
import sys
import os
import random

from quantifydrivers import machine_learning

def reset_seeds(g,seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)

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


def build_datasets_and_loaders(configuration, seed, generator):
    print("seed: ", seed, " CONF seed: ", configuration["SEED"])

    seed = configuration["SEED"]

    percentile = configuration["percentile_to_load"]
    paths = load_mock_paths(configuration["paths"]["base_folder"], configuration["SITE"], percentile)

    file_local_scale = paths["local"]
    file_g500 = paths["g500"]
    file_g200 = paths["g200"]
    file_psl = paths["psl"]
    file_co2 = paths["co2"]
    variables_era5 = configuration["dataset_config"]["variables_era5"]
    variables_era5land = configuration["dataset_config"]["variables_era5land"]
    start_date = configuration["dataset_config"]["start_date"]

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

    SITE_HYPMS_fixed = {
        'belgrado': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                     'nonextreme_weights_ctt': 1},
        'hannover': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                     'nonextreme_weights_ctt': 1},
        'stockholm': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                      'nonextreme_weights_ctt': 1},
        'lyon': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
        'cordoba': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                    'nonextreme_weights_ctt': 1},
        'marrakech': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                      'nonextreme_weights_ctt': 1}}

    # Create empty dictionary with the base HYPMS
    SITE_HYPMS = SITE_HYPMS_fixed.copy()

    print(f"Loading hyperparameters for percentile: {percentile}")
    params = load_hypms_from_file(configuration["SITE"], percentile=percentile, file_name="file_with_hypms.txt")
    if params:
        SITE_HYPMS[configuration["SITE"]] = params
    print(f"Loaded hyperparameters for {configuration["SITE"]}: {SITE_HYPMS[configuration["SITE"]]}")

    print(f"Doing site: {configuration["SITE"]}")
    print(f"*** Setting up file paths for site: {configuration["SITE"]} ***")  # NEW PRINT

    print("seed: ", seed, " CONF seed: ", configuration["SEED"])

    reset_seeds(generator,seed)

    print("seed: ", seed, " CONF seed: ", configuration["SEED"])

    # -------------------------------
    # 1. CREATE LOCAL-SCALE DATASETS
    # -------------------------------

    train_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local_scale, file_CO2=file_co2, **era5land_train_conf)

    test_dataset = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local_scale, file_CO2=file_co2, **era5land_test_conf)


    # --------------------------------
    # 2. CREATE LARGE-SCALE DATASETS
    # --------------------------------

    train_features_era5 = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **era5_train_conf)

    test_features_era5 = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **era5_test_conf)

    dataloader_conf = dict(
        batch_size=SITE_HYPMS[configuration["SITE"]]['batch_size'],
        drop_last=False,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    dataloader_test_conf = dict(
        batch_size=SITE_HYPMS[configuration["SITE"]]['batch_size'],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )


    # ---------------------------------------
    # 3. COMBINED DATASET
    # ---------------------------------------
    combined_train = machine_learning.CombinedDataset(train_dataset, train_features_era5, variables=variables_era5)
    combined_test = machine_learning.CombinedDataset(test_dataset, test_features_era5, variables=variables_era5)


    # ---------------------------------------
    # 4. TRAIN/VAL SPLIT
    # ---------------------------------------

    train_size = int(0.8 * len(combined_train))
    val_size = len(combined_train) - train_size

    train_subset, val_subset = random_split(combined_train, [train_size, val_size], generator=generator)


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
