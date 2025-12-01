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
# HELPERS
# ----------------------------------------------------------

def load_hypms_from_file(site_name, percentile='90p', base_path='/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs',file_name=None):
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
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None

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
                    
    return hypms

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
# CONFIGURATION
# ----------------------------------------------------------

SITES = ['cordoba', 'hannover', 'lyon', 'stockholm', 'belgrado','marrakech']
PERCENTILE = "90p"

RAW_LOCAL_TEMPLATE = "/gpfs/scratch/bsc32/bsc167965/data/era5_land/lagged_anomalies_and_event_detection/{percentile}_{site}_lagged_standarized_anomalies_and_extreme_detection.nc"
FILE_CO2 = "/home/bsc/bsc167965/TFM/ML/data_files/daily_co2_JJA.nc"

FILE_G500 = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/g500_1x1_lagged_standarized_anomalies.nc"
FILE_G500 = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/g200_1x1_lagged_standarized_anomalies.nc"
FILE_PSL  = "/gpfs/scratch/bsc32/bsc167965/data/era5/lagged_anomalies/psl_1x1_lagged_standarized_anomalies.nc"

START_DATE = "1950-01-01"
TRAIN_END  = "2013-12-31"
TEST_START = "2014-01-01"
TEST_END   = "2023-12-31"

variables_era5land = ['swvl1','swvl2','swvl3']
variables_era5     = ['g500','g200','psl']


SITE_HYPMS_fixed = {'belgrado':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1},
             'hannover':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'stockholm':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'lyon':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'cordoba':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}, 
             'marrakech':{'lr':1e-4,'w_decay':0.01,'batch_size':32,'extreme_weights_ctt':1,'nonextreme_weights_ctt':1}}

SITE_HYPMS = SITE_HYPMS_fixed.copy()

print(f"Loading hyperparameters for percentile: {PERCENTILE}")
for site_name in SITES:
    params = load_hypms_from_file(site_name, percentile=PERCENTILE,file_name = "file_with_hypms.txt")
    if params:
        SITE_HYPMS[site_name] = params 
    print(f"Loaded hyperparameters for {site_name}: {SITE_HYPMS[site_name]}")




# ----------------------------------------------------------
# MAIN EXTRACTION
# ----------------------------------------------------------

def build_dataset_for_site(site):
    print(f"\n=== Extracting site: {site} ===")

    out_path = os.path.join("datasets", site, PERCENTILE)
    os.makedirs(out_path, exist_ok=True)

    print(f"Doing site: {site}")
    local_path = RAW_LOCAL_TEMPLATE.format(percentile=PERCENTILE, site=site)

    
        
    # Datasets configuration dictionaries --------------------------------------------------

    # Start date for all datasets
    start_date = "1950-01-01"
    # Variables large-scale and local scale to use 
    variables_era5 = ['g500','g200','psl'] 
    variables_era5land = ['swvl1','swvl2','swvl3']  
    
    # Local-scale datasets configuration
    _ERA5LAND_TRAIN_DATASET_CONF = dict(
    start_date= start_date,
    end_date= "2013-12-31",
    months = [6,7,8],
    variables = variables_era5land
        )

    _ERA5LAND_TEST_DATASET_CONF = dict(
    start_date= "2014-01-01",
    end_date= "2023-12-31",
    months = [6,7,8],
    variables = variables_era5land
        )

    # Large-scale datasets configuration
    _ERA5_TRAIN_DATASET_CONF = dict(
    start_date= start_date,
    end_date= "2013-12-31",
    months = [6,7,8],
    start_lag = 1,
    lags_era5 = 1,
    variables = variables_era5
        )

    _ERA5_TEST_DATASET_CONF = dict(
    start_date= "2014-01-01",
    end_date= "2023-12-31",
    months = [6,7,8],
    start_lag = 1,
    lags_era5 = 1,
    variables = variables_era5      
        )

    
    # Datasets ERA5land data --------------------------------------------------------------
    
    train_dataset = LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(file_path=local_path, file_CO2=FILE_CO2 , **_ERA5LAND_TRAIN_DATASET_CONF)
    test_dataset = LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(file_path=local_path, file_CO2=FILE_CO2 , **_ERA5LAND_TEST_DATASET_CONF)
    
    # Datasets ERA5 data ------------------------------------------------------------------

    train_features_era5 = LargeScale_Dataset_extremes(FILE_G500,FILE_G500,FILE_PSL, **_ERA5_TRAIN_DATASET_CONF) # shape: features, time, lat, lon 
    test_features_era5 = LargeScale_Dataset_extremes(FILE_G500,FILE_G500,FILE_PSL, **_ERA5_TEST_DATASET_CONF)
    
   
    combined_train_dataset  = CombinedDataset(train_dataset,train_features_era5, variables=variables_era5)
    combined_test_dataset = CombinedDataset(test_dataset,test_features_era5, variables=variables_era5)





if __name__ == "__main__":
    for site in SITES:
        build_dataset_for_site(site)













