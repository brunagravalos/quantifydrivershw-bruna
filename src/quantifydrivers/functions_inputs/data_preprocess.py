
# =========================================================================================
# IMPORT NEEDED PACKAGES
# ========================================================================================

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset as ncread
import xarray as xr 
from scipy.stats import linregress
from datetime import datetime, timedelta
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import os
import xesmf as xe#for regridding 

# =========================================================================================


# =========================================================================================
# This script provides utilities for preparing climate datasets for machine learning 
# and spatial analysis.
#
# Main functionalities:
#   - Create lagged features:
#       * create_lagged_features_single: Generate lagged anomalies for a single variable.
#       * create_lagged_features_multiple: Generate lagged versions for multiple variables 
#         using a lag dictionary.
#   - Regridding:
#       * regridded_dataset: Regrid data to a 1°x1° grid using bilinear interpolation (via xesmf).
#
# Typical use cases:
#   - Preparing ERA5 or climate reanalysis datasets for training artificial neural networks (ANNs).
#   - Standardizing spatial resolution across datasets.
#
# Dependencies: numpy, xarray, pandas, matplotlib, cartopy, netCDF4, scipy, xesmf, os, datetime
# =========================================================================================


# Lagged-data creation ------------------------------------------------------------------------------------------------------

def create_lagged_features_single(dataset, var, lags, new_prefix):

    """Create lagged time series for a variable.
    In:
    dataset (xarray): Input dataset containing the variable.
    var (str): Name of the variable to create lags for.
    lags (list): List of integers representing the lag steps (e.g., [1,2,3,4,5,6,7]).
    new_prefix (str): Prefix to add to the lagged variable names (e.g., 'era5_land_').
    Out:
    dataset: Dataset with original variable + lagged variables.
    """

    lags = sorted(lags, reverse=True)
    new_vars = {}
    
    for lag in lags:
        var_name = f"{new_prefix}{var}_anomalies_lag{lag}"
        try:
            new_var = dataset[f'{var}_anomalies'].shift(time=lag)
        except KeyError:
            new_var = dataset[f'{var}'].shift(time=lag)
        new_var.attrs = {
            'long_name': f'{lags} lags {var}_anomalies lag number {lag}',
            'description': (
                f"Lag number {lag} for a total lag of {lags} for variable {var}_anomalies. "
                f"Part of lagged-data for ERA5 input to ANN."
            )
        }
        new_vars[var_name] = new_var
    
    return xr.Dataset(new_vars, attrs=dataset.attrs)


#For multiple variables -------------------------------------------------------------------------------------------------------

def create_lagged_features_multiple(dataset, variables, lag_dict, new_prefix):
    """
    Create lagged time series for multiple variables with specified lags.
    
    In:
        dataset (xarray): Input dataset containing variables.
        variables (list): List of variable names to create lags for.
        lag_dict (dict): Dictionary mapping variable names to their respective lag lists.
                          Example: {'tasmax': [1,2,3], 'swvl1': [1,2,3,4,5,6,7]}.
        new_prefix (str): Prefix to add to the lagged variable names (e.g., 'era5_land_').
    
    Out:
        dataset: Dataset with original variables + lagged variables.
    """

    dataset_save = dataset.copy()
    
    # Iterate over each variable to process
    for var in variables:
        # Get the lags for this variable from the dictionary
        lags = lag_dict.get(f'{var}', [])
        
        # Create lagged variables for each lag
        for lag in lags:
            # Shift the variable by `lag` timesteps and add to the dataset
            dataset_save[f"{new_prefix}{var}_lag{lag}"] = dataset_save[f'{var}'].shift(time=lag)
            dataset_save[f"{new_prefix}{var}_lag{lag}"].attrs = {
            'long name':f'{lags} lags {var} lag number {lag}',
            'description': (f"Lag numer {lag} for a toal lag of {lags} for variable {var}. Part of lagged-data for ERA5 land that will serve as input for a ANN. The ANN will target a certain day in a certain region and ERA5 land data will be provided for the specific point." )
        }

    return dataset_save


# -------------------------------------------------------------------------------------------------------------------------------------------


#Regridder for ERA5 data -------------------------------------------------------------------------------------------------

def regridded_dataset(ds,variable):

    """ 
    Regrid an xarray Dataset to a 1°x1° grid using bilinear interpolation.
    In:
        ds (xarray.Dataset): Input dataset with 'lat' and 'lon' coordinates.
        variable (str): Name of the variable to regrid.
    Out:
        ds_regridded (xarray.Dataset): Regridded dataset with 1°x1° resolution.
    Notes:
        - Requires xesmf package for regridding.
        - Preserves global and variable attributes.
        """

    # Extract the latitude and longitude bounds from the dataset
    lat_min, lat_max = ds.lat.min().item(), ds.lat.max().item()
    lon_min, lon_max = ds.lon.min().item(), ds.lon.max().item()
    
    # Create the new target grid with 1° resolution
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(lat_min, lat_max + 1, 1)),  # Ensure it covers full range
            "lon": (["lon"], np.arange(lon_min, lon_max + 1, 1)),
        }
    )
    
    # Create the regridder
    regridder = xe.Regridder(ds, ds_out, method="bilinear")  # You can also try "nearest_s2d" or "conservative"
    
    # Apply regridding
    ds_regridded = regridder(ds)
    
    # Preserve global attributes
    ds_regridded.attrs = ds.attrs  
    
    # Preserve variable attributes
    ds_regridded[variable].attrs = ds[variable].attrs 

    return ds_regridded

# ====================================================================================================================










    