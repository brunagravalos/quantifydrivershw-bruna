# data_loading.py
# ============================================================
# 0. DETERMINISM BLOCK (must be FIRST, before importing torch)
# ============================================================

import os


import random
import numpy as np
import torch

import yaml
from torch.utils.data import DataLoader, random_split

import sys



from quantifydrivers import machine_learning


def check_seeds():
    print("--- CURRENT SEED STATES (from data_loading)---")
    print(f"Torch seed: {torch.initial_seed()}")
    print(f"NumPy seed: {np.random.get_state()[1][0]}")
    print(f"Python random seed: {random.getstate()[1][0]}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")


def load_hypms_from_file(site_name, percentile='90p', base_path='/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs',
                         file_name=None):
    """
    Loads hyperparameters for a given site and percentile from a text file. The hyperparameters to load are hardcoded.
    """
    hypms = {}

    # 1. Get the directory of the current script:
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Path to 'quantifydrivers' directory
    quantifydrivers_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Construct the final path using os.path.join for reliability
    file_path = os.path.join(
        quantifydrivers_dir,
        "data_files",
        "HYPMS_optimization_results",
        f"g500_1lag_{site_name}_best_params_{percentile}_with_testing_phase.txt"
    )
    print(f"*** HYPMs: Checking file path: {file_path} ***")

    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None

    print(f"*** HYPMs: File found. Parsing contents... ***")

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

    print(f"*** HYPMs: Successfully loaded {len(hypms)} hyperparameters. ***")
    return hypms


# -------------------------------------------------------
# Helper: load paths for (mock structure) dataset
# -------------------------------------------------------
def load_mock_paths(base_folder: str, site: str, percentile: str):
    return {
        "g500": os.path.join(base_folder, "mockLargeScale_data", "file_g500.nc"),
        "g200": os.path.join(base_folder, "mockLargeScale_data", "file_g200.nc"),
        "psl": os.path.join(base_folder, "mockLargeScale_data", "file_psl.nc"),
        "co2": os.path.join(base_folder, "mockLargeScale_data", "file_CO2.nc"),
        "local": os.path.join(base_folder, "mockLocalScale_data", f"file_local_{percentile}_{site}.nc")
    }


# -------------------------------------------------------
# Main entry point called by training script
# -------------------------------------------------------
def load_datasets_and_loaders(config_path: str, g):
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
    print(f"\n*** Starting data loading pipeline with config: {config_path} ***")

    # ------------------------------
    # 1. Read YAML config
    # ------------------------------
    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    SITE = CONF["SITE"]
    SEED = CONF["SEED"]


    percentile = CONF["percentile_to_load"]
    print(f"*** Config loaded: Site={SITE}, Seed={SEED}, Percentile={percentile} ***")
    #set_global_determinism(seed=SEED)

    check_seeds()

    # ------------------------------
    # 2. Resolve file paths
    # ------------------------------
    paths = load_mock_paths(CONF["paths"]["base_folder"], SITE, percentile)

    file_local = paths["local"]
    file_g500 = paths["g500"]
    file_g200 = paths["g200"]
    file_psl = paths["psl"]
    file_CO2 = paths["co2"]
    print(f"*** Local file path: {file_local} ***")

    variables_era5 = CONF["dataset_config"]["variables_era5"]
    variables_era5land = CONF["dataset_config"]["variables_era5land"]
    start_date = CONF["dataset_config"]["start_date"]

    # which script should this go to?
    params = load_hypms_from_file(SITE, percentile=percentile, file_name="file_with_hypms.txt")

    if params:
        CONF['site_hypms'] = params
    print(f"Loaded hyperparameters for {SITE}: {CONF['site_hypms']}", type(CONF['site_hypms']["lr"]))

    print(f"Doing site: {SITE}")

    # ------------------------------
    # 3. Build dataset configs
    # ------------------------------
    print("*** Building dataset configuration dictionaries ***")
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
    print("*** Creating Local-Scale (NN) Datasets ***")
    train_local = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local, file_CO2=file_CO2, **ERA5LAND_TRAIN
    )
    print(f"*** Train Local Dataset size: {len(train_local)} ***")

    test_local = machine_learning.LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=file_local, file_CO2=file_CO2, **ERA5LAND_TEST
    )
    print(f"*** Test Local Dataset size: {len(test_local)} ***")

    print("*** Creating Large-Scale (CNN) Datasets ***")
    train_large = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **ERA5_TRAIN
    )
    print(f"*** Train Large Dataset size: {len(train_large)} ***")

    test_large = machine_learning.LargeScale_Dataset_extremes(
        file_g500, file_g200, file_psl, **ERA5_TEST
    )
    print(f"*** Test Large Dataset size: {len(test_large)} ***")

    # ------------------------------
    # 5. Combined dataset
    # ------------------------------
    print("*** Creating Combined Datasets ***")
    combined_train = machine_learning.CombinedDataset(train_local, train_large, variables=variables_era5)
    combined_test = machine_learning.CombinedDataset(test_local, test_large, variables=variables_era5)
    print(f"*** Combined Train Dataset size: {len(combined_train)} ***")

    # ------------------------------
    # 6. Train/val split
    # ------------------------------
    g = torch.Generator()
    g.manual_seed(SEED)

    train_size = int(0.8 * len(combined_train))
    val_size = len(combined_train) - train_size
    print(f"*** Splitting Combined Train: Train={train_size}, Validation={val_size} ***")

    train_subset, val_subset = random_split(combined_train, [train_size, val_size], generator=g)
    print("*** Train/Validation split complete. ***")

    # ------------------------------
    # 7. Create dataloaders
    # ------------------------------
    dl_conf = dict(
        batch_size=CONF['site_hypms']['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=g
    )

    dl_test_conf = dict(
        batch_size=CONF['site_hypms']['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    print(f"*** Creating DataLoaders with batch_size: {CONF['site_hypms']['batch_size']} ***")

    loaders = {
        "train": DataLoader(train_subset, **dl_conf),
        "val": DataLoader(val_subset, **dl_conf),
        "test": DataLoader(combined_test, **dl_test_conf),
    }

    datasets = {
        "train_local": train_local,
        "test_local": test_local,
        "train_large": train_large,
        "test_large": test_large,
        "combined_train": combined_train,
        "combined_test": combined_test,
        "train_subset": train_subset,
        "val_subset": val_subset,
    }

    print(f"*** All DataLoaders initialized. ***")
    print(f"\nSaving processed config back to: {config_path}")

    # Ensure numeric types are correct
    for key in ["lr", "w_decay", "minority_weight_multiplier", "batch_size"]:
        if key in CONF["site_hypms"]:
            try:
                CONF["site_hypms"][key] = float(CONF["site_hypms"][key])
            except ValueError:
                pass

    with open(config_path, "w") as f:
        yaml.safe_dump(CONF, f, sort_keys=False)

    print(f"*** Config file updated and saved successfully. Data loading complete. ***")
    return datasets, loaders