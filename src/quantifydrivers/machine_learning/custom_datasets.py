
# =============================================================================================================================
# Custom Datasets -------------------------------------------------------------------------------------------------------------
# =============================================================================================================================

# =============================================================================================================================
# Import necessary libraries
# =============================================================================================================================

import torch
import scipy 
import xarray as xr
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import optuna 
import random
from tqdm import tqdm
import torch.nn as nn                   
import torch.nn.functional as F


def open_xr_dataset(path: str) -> xr.Dataset:
    """
    Open NetCDF or Zarr transparently.
    """
    if path.endswith(".nc"):
        return xr.open_dataset(path)
    else:
        # Zarr group or store
        return xr.open_zarr(path, consolidated=True)


# =============================================================================================================================
# This script defines custom PyTorch Dataset classes for handling ERA5 and ERA5-Land climate data,
# tailored for extreme event prediction tasks.
#
# Main classes:
#   - CombinedDataset: Combines local-scale and large-scale datasets while ensuring temporal coherence.
#   - LargeScale_Dataset_extremes: Loads large-scale ERA5 predictors (g500, g200, psl) with lagged anomalies.
#   - Dataset_count_observational_TX: Handles observational ERA5-Land data (single point) with extremes.
#   - LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2: Uses averaged soil moisture anomalies
#       and CO2 concentration as predictors for extreme classification.
#   - SPEI_extremes_location_dataset: Combines CO2 concentration and drought indices (SPEI/SPI)
#       as predictors for extreme classification.
#
# Features:
#   - Supports lagged variables for temporal dependencies.
#   - Integrates CO2 and drought indicators as additional predictors.
#   - Handles temporal filtering by date and month.
#   - Removes NaN samples for clean training input.
#   - Returns tensors compatible with PyTorch DataLoader.
#
# Dependencies: torch, numpy, xarray
# =============================================================================================================================


# Combined local-scale and large-scale dataset --------------------------------------------------------------------------------

class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self, local_data, large_data, variables):
        
        self.variables = variables  # Store selected variables

        self.local_data = local_data
        self.large_data = large_data

        self._test_coherence()
        
        
    def _test_coherence(self):
        """
        Check if the local and large data have the same time values
        """
        for t in ["day", "month", "year"]: 
            for v in ["ds_g200", "ds_g500", "ds_psl"]:
                if v.split('_')[1] in self.variables:
                    _times1 = getattr(self.local_data.ds.time.dt, t).values
                    _times2 = getattr(self.large_data, v).time.dt
                    _times2 = getattr(_times2, t).values
                            
                    assert (_times1 == _times2).all(), f"Time values do not match for {t} in {v}"
        
    def __len__(self):
        return len(self.local_data.labels)
    
    def __getitem__(self, idx):
        
        local_features = self.local_data[idx][0]
        large_features = self.large_data[idx]
        local_labels = self.local_data[idx][1]
        
        return local_features, large_features, local_labels 

# --------------------------------------------------------------------------------------------------------------------
# Large-Scale dataset -------------------------------------------------------------------------------------------------

class LargeScale_Dataset_extremes(Dataset):
    def __init__(self, file_g500, file_g200, file_psl, start_date, end_date, months, start_lag, lags_era5, variables, transform=None):
        """
        Args:
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            start_lag (int) : 1 for the lag 1. 
            lags_era5 (int) : number of lags to take in the dataset
            months (list of int): List of months to filter.
            variables (list of str): Variables to use (subset of ['g500','g200','psl']).
            transform (callable, optional): Optional transform to be applied.
        """

        self.variables = variables  # Store selected variables

        # Prepare a dict to hold datasets and features
        datasets = {}
        lagged_vars_dict = {}

        # g500
        if "g500" in self.variables:
            ds_g500 = open_xr_dataset(file_g500).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
            ds_g500 = ds_g500.sel(time=ds_g500.time.dt.month.isin(months), drop=True)
            datasets["g500"] = ds_g500
            lagged_vars_dict["g500"] = [f'lagged_era5g500_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1)]

        # g200
        if "g200" in self.variables:
            ds_g200 = open_xr_dataset(file_g200).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
            ds_g200 = ds_g200.sel(time=ds_g200.time.dt.month.isin(months), drop=True)
            ds_g200 = ds_g200.squeeze('plev', drop=True) 
            datasets["g200"] = ds_g200
            lagged_vars_dict["g200"] = [f'lagged_era5g200_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1)]

        # psl
        if "psl" in self.variables:
            ds_psl = open_xr_dataset(file_psl).sel(time=slice(start_date, end_date)).sel(lon=slice(-54,69))
            ds_psl = ds_psl.sel(time=ds_psl.time.dt.month.isin(months), drop=True)
            datasets["psl"] = ds_psl
            lagged_vars_dict["psl"] = [f'lagged_era5psl_anomalies_lag{lag}' for lag in range(start_lag, lags_era5+1)]

        for key, ds in datasets.items():
            setattr(self, f"ds_{key}", ds)

        # keep the dict 
        self.datasets = datasets

        self.lagged_vars = lagged_vars_dict
        self.all_features = [var for vars_list in lagged_vars_dict.values() for var in vars_list]

        # Convert datasets to arrays in the same shape and concatenate
        feature_arrays = []
        for var_name in self.variables:
            arr = datasets[var_name][self.lagged_vars[var_name]].to_array().transpose('time', 'variable', 'lat', 'lon').values
            feature_arrays.append(arr)

        self.features = np.concatenate(feature_arrays, axis=1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.features[idx], dtype=torch.float32)
        return sample

# --------------------------------------------------------------------------------------------------------------
# Dataset only for counting observational data-------------------------------------------------------------------------------------------------------

class Dataset_count_observational_TX(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, start_date, end_date, months,scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) #open selected months 
        
        self.labels = self.ds['tx_mean_extreme_classification'].values


    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() 

    
# ------------------------------------------------------------------------------------------------------------------------------
# Dataset for ERA5Land including CO2 concentration and lagged soil moisture averaged in time

class LocalScale_Dataset_extremes_location_swvl_averaged_including_CO2(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point)."""

    def __init__(self, file_path, file_CO2, start_date, end_date, months, variables, scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """

        self.variables = variables  # Store selected variables

        # Load dataset with lagged-data and extreme classification 
        self.ds = open_xr_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True)
        self.dsco2 = open_xr_dataset(file_CO2).sel(time=slice(start_date, end_date))
        self.co2conc = self.dsco2['co2_concentration'].values

        # Define all lagged swvl variables
        all_lagged_vars = {
            'swvl1': [f'lagged_era5_land_swvl1_anomalies_lag{i}' for i in range(1, 8)],
            'swvl2': [f'lagged_era5_land_swvl2_anomalies_lag{i}' for i in range(1, 8)],
            'swvl3': [f'lagged_era5_land_swvl3_anomalies_lag{i}' for i in range(1, 8)],
        }

        # Keep only the ones selected
        self.lagged_vars = {k: v for k, v in all_lagged_vars.items() if k in self.variables}

        # Collect feature names
        self.all_features = ['co2'] + [var for var in self.lagged_vars.keys()] 

        # Extract features (lagged-data in each time-step) and labels(exteme/non-extreme)

        swvl_avg_features = []
        
        for key in self.lagged_vars:
            swvl_lags = self.ds[self.lagged_vars[key]].to_array(dim='lag').transpose('time', 'lag').values
            swvl_avg = np.nanmean(swvl_lags, axis=1)  
            swvl_avg_features.append(swvl_avg)       
        
        self.features_loc = np.stack(swvl_avg_features, axis=1)
        
        self.co2conc = self.co2conc.reshape(-1,1)
        self.features = np.concatenate([self.co2conc,self.features_loc],axis=1)
    
        self.labels = self.ds['tasmax_extreme_classification'].values

        # Remove NaN values from samples 
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]


    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                      
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() 

    
# -----------------------------------------------------------------------------------------------------------------------------
# Dataset for ERA5Land including CO2 concentration and spei/spi index

class SPEI_extremes_location_dataset(Dataset):
    """Custom Dataset for ERA5 Land Data (Single Point). Uses spei/spi results, CO2 concentration, and extreme classification labels. 
    There is no need for lagged features in the case of using spei or spi since its itself a lagged variable, because it is calculated based 
    on the previous months' precipitation and temperature data (temeprature if spei is computed).
    
    SPEI can be computed with three main methods depending on the data availability:
    
    """

    def __init__(self, file_path, file_CO2, files_spei, start_date, end_date, months, spei_variables, num_lags, scaler=None):
        """
        Args:
            file_path (str): Path to the NetCDF file containing the ERA5 land dataset.
            file_CO2 (str): Path to the NetCDF file containing CO2 concentration data.
            file_spei (str): Path to the NetCDF file containing SPEI data.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list int): List months to filter 
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
        """
        # Load dataset with lagged-data and extreme classification 
        self.ds = xr.open_dataset(file_path).sel(time=slice(start_date, end_date))
        self.ds = self.ds.sel(time=self.ds.time.dt.month.isin(months),drop=True) 

        # spei data
        if len(files_spei) > 1:
            self.dsspei1 = xr.open_dataset(files_spei[0]).sel(time=slice(start_date, end_date))
            self.dsspei1 = self.dsspei1.sel(time=self.dsspei1.time.dt.month.isin(months),drop=True)

            self.dsspei2 = xr.open_dataset(files_spei[1]).sel(time=slice(start_date, end_date))
            self.dsspei2 = self.dsspei2.sel(time=self.dsspei2.time.dt.month.isin(months),drop=True)
        else:
            self.dsspei1 = xr.open_dataset(files_spei[0]).sel(time=slice(start_date, end_date))
            self.dsspei1 = self.dsspei1.sel(time=self.dsspei1.time.dt.month.isin(months),drop=True)
            self.dsspei2 = None

        self.dsco2 = xr.open_dataset(file_CO2).sel(time=slice(start_date, end_date))
        self.co2conc = self.dsco2['co2_concentration'].values


        # Collect feature names
        self.all_features = ['co2'] + [spei_variable for spei_variable in spei_variables] 

        # Extract features (lagged-data in each time-step) and labels(exteme/non-extreme)

        if len(spei_variables) > 1:
            self.spei_data1 = self.dsspei1[spei_variables[0]].values.reshape(-1,1)
            self.spei_data2 = self.dsspei2[spei_variables[1]].values.reshape(-1,1)
            self.spei_data = np.concatenate([self.spei_data1,self.spei_data2], axis=1)
        else:
            self.spei_data = self.dsspei[spei_variable].values.reshape(-1,1)
    
        # Reshape data to ensure it is 2D (samples, features)
        self.co2conc = self.co2conc.reshape(-1,1)
        # Concatenate CO2 concentration and SPEI data along the feature axis
        self.features = np.concatenate([self.co2conc,self.spei_data],axis=1)
    
        self.labels = self.ds['tasmax_extreme_classification'].values

        # Remove NaN values from samples 
        valid_indices = ~np.isnan(self.features).any(axis=1)
        self.features = self.features[valid_indices]
        self.labels = self.labels[valid_indices]

    def __len__(self):
        """In our case, the number of time steps in each feature."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a selected sample from the dataset using the idx indexes"""                     
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze() 

# =============================================================================================================================