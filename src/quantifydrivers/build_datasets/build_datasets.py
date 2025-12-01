"""
build_datasets.py

Extracts preprocessed datasets (local-scale, large-scale, labels) for all sites,
across train / val / test splits. Saves everything as .npz and .npy files,
clean, small, and reproducible.

This script uses the existing dataset classes from machine_learning.
"""

import os
import json
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from machine_learning import (
    LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2,
    LargeScale_Dataset_extremes,
    CombinedDataset
)

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

SITES = ['cordoba', 'hannover', 'lyon', 'stockholm', 'belgrado','marrakech']
PERCENTILE = "90p"

RAW_LOCAL_TEMPLATE = "/gpfs/scratch/bsc32/bsc167965/data/era5_land/lagged_anomalies_and_event_detection/{percentile}_{site}_lagged_standarized_anomalies_and_extreme_detection.nc"
FILE_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"

FILE_G500 = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/g500_1x1_lagged_standarized_anomalies.nc"
FILE_G200 = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/g200_1x1_lagged_standarized_anomalies.nc"
FILE_PSL  = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/psl_1x1_lagged_standarized_anomalies.nc"

START_DATE = "1950-01-01"
TRAIN_END  = "2013-12-31"
TEST_START = "2014-01-01"
TEST_END   = "2023-12-31"

variables_era5land = ['swvl1','swvl2','swvl3']
variables_era5     = ['g500','g200','psl']


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def extract_loader(dataset, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    local_data = []
    large_data = []
    labels     = []

    for local_x, large_x, y in loader:
        local_data.append(local_x.squeeze(0).numpy())
        large_data.append(large_x.squeeze(0).numpy())
        labels.append(y.item())

    return (
        np.stack(local_data),
        np.stack(large_data),
        np.array(labels)
    )


def save_npz_split(out_path, name, local, large, labels):
    np.savez(os.path.join(out_path, f"{name}_local.npz"),
             data=local,
             variables=variables_era5land + ["CO2"])
    np.savez(os.path.join(out_path, f"{name}_large.npz"),
             data=large,
             variables=variables_era5,
             num_lags=1)
    np.save(os.path.join(out_path, f"{name}_labels.npy"), labels)


# ----------------------------------------------------------
# MAIN EXTRACTION
# ----------------------------------------------------------

def build_dataset_for_site(site):
    print(f"\n=== Extracting site: {site} ===")

    out_path = os.path.join("datasets", site, PERCENTILE)
    os.makedirs(out_path, exist_ok=True)

    # ---------------------------------------------
    # Instantiate LOCAL datasets
    # ---------------------------------------------
    local_path = RAW_LOCAL_TEMPLATE.format(percentile=PERCENTILE, site=site)

    train_local = LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=local_path,
        file_CO2=FILE_CO2,
        start_date=START_DATE,
        end_date=TRAIN_END,
        months=[6,7,8],
        variables=variables_era5land
    )

    test_local = LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(
        file_path=local_path,
        file_CO2=FILE_CO2,
        start_date=TEST_START,
        end_date=TEST_END,
        months=[6,7,8],
        variables=variables_era5land
    )

    # ---------------------------------------------
    # Instantiate LARGE-SCALE datasets
    # ---------------------------------------------
    train_large = LargeScale_Dataset_extremes(
        FILE_G500, FILE_G200, FILE_PSL,
        start_date=START_DATE,
        end_date=TRAIN_END,
        months=[6,7,8],
        start_lag=1,
        lags_era5=1,
        variables=variables_era5
    )

    test_large = LargeScale_Dataset_extremes(
        FILE_G500, FILE_G200, FILE_PSL,
        start_date=TEST_START,
        end_date=TEST_END,
        months=[6,7,8],
        start_lag=1,
        lags_era5=1,
        variables=variables_era5
    )

    # ---------------------------------------------
    # Combined dataset (used only for splitting)
    # ---------------------------------------------
    combined_train = CombinedDataset(train_local, train_large, variables=variables_era5)

    train_size = int(len(combined_train) * 0.8)
    val_size   = len(combined_train) - train_size
    train_subset, val_subset = random_split(combined_train, [train_size, val_size])

    test_combined = CombinedDataset(test_local, test_large, variables=variables_era5)

    # ---------------------------------------------
    # Extract ALL SPLITS
    # ---------------------------------------------
    print("- Extracting TRAIN...")
    tr_local, tr_large, tr_y = extract_loader(train_subset)

    print("- Extracting VAL...")
    va_local, va_large, va_y = extract_loader(val_subset)

    print("- Extracting TEST...")
    te_local, te_large, te_y = extract_loader(test_combined)

    # ---------------------------------------------
    # Save all splits
    # ---------------------------------------------
    save_npz_split(out_path, "train", tr_local, tr_large, tr_y)
    save_npz_split(out_path, "val", va_local, va_large, va_y)
    save_npz_split(out_path, "test", te_local, te_large, te_y)

    # ---------------------------------------------
    # Metadata
    # ---------------------------------------------
    metadata = {
        "site": site,
        "percentile": PERCENTILE,
        "variables_local": variables_era5land + ["CO2"],
        "variables_large": variables_era5,
        "lags": 1,
        "splits": ["train", "val", "test"]
    }

    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved dataset for {site} in {out_path}")


if __name__ == "__main__":
    for site in SITES:
        build_dataset_for_site(site)
