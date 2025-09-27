
# =========================================================================================================================
# IMPORT NEEDED PACKAGES
# =========================================================================================================================

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
from . import loess

# ========================================================================================================================

# =========================================================================================================================
# This script provides a set of functions to compute climate-related indices and detect extreme heatwave events.
# 
# Main functionalities:
#   - Compute climatologies (raw, moving-window, and LOESS-smoothed).
#   - Calculate percentiles, anomalies, and standardized anomalies.
#
# Dependencies: numpy, xarray, pandas, matplotlib, cartopy, scipy, netCDF4, and custom loess.
# =========================================================================================================================

#Function for only computing the cliamtology -----------------------------------------------------------------------------

def Compute_climatology(dataset,variable,window,date1,date2):

    '''climatology: computed by single days 
    clim_window: computed by averaging moving 5 day wondow over time time period
    percentile: nth percentile computed for a moving m-day window. Percentile calculated per day and then averaged over a 5-day windw'''
    ds = dataset.sel(time=slice(date1,date2))
    climatology = ds[variable].groupby('time.dayofyear').mean('time') 

    #loess fit for the climatology -----------

    loess_climatology = xr.apply_ufunc(
        loess.loess_ts, 
        climatology,  # Your DataArray
        input_core_dims=[["dayofyear"]],  
        output_core_dims=[["dayofyear"]],  
        vectorize=True,  
        dask="parallelized",  
        kwargs={"na_rm": True, "window": 30, "degree": 1}  
    )

    # ---------------------------------------

    window_series = dataset[variable].rolling(time=window,center=True,min_periods=1).mean(skipna=True) 
    clim_window = window_series.groupby('time.dayofyear').mean('time') 

    return climatology,clim_window, window_series, loess_climatology

# ------------------------------------------------------------------------------------------------------------------------------


#Function for computing climatology and smothed percentile using a m-day window. 

def Compute_window_percentile_reference_period(dataset,variable,quantile_value,window,date1,date2):
    '''climatology: computed by single days 
    clim_window: computed by averaging moving 5 day wondow over time time period
    percentile: nth percentile computed for a moving m-day window. Percentile calculated per day and then averaged over a 5-day windw'''
    ds = dataset.sel(time=slice(date1,date2))
    climatology = ds[variable].groupby('time.dayofyear').mean('time') 

    #loess fit for the climatology -----------

    loess_climatology = xr.apply_ufunc(
        loess.loess_ts, 
        climatology,  
        input_core_dims=[["dayofyear"]],
        output_core_dims=[["dayofyear"]], 
        vectorize=True,  
        dask="parallelized",  
        kwargs={"na_rm": True, "window": 30, "degree": 1} 
    )

    #percentile computation 
    ds_rolling = ds[variable].rolling(time=5,center=True).construct("window")
    ds_rolling_grouped = ds_rolling.groupby('time.dayofyear')
    percentile = ds_rolling_grouped.quantile(quantile_value,dim=('time','window'))

    #window series and climatology of window series 

    window_series = dataset[variable].rolling(time=window,center=True,min_periods=1).mean(skipna=True) 
    clim_window = window_series.groupby('time.dayofyear').mean('time') 

    return climatology,percentile,clim_window, window_series, loess_climatology

# ------------------------------------------------------------------------------------------------------------------------------

#Function for standarized anomalies. Two options, raw or window smoothing

def Compute_anomalies(dataset, variable, date1, date2):
    """
    Computes the anomalies for a given variable for two climatology computations:
    - Raw climatology (simple mean per day over the whole period).
    - Climatology computed using a moving m-day window.

    Also standardizes the anomalies to have a mean of 0 and a standard deviation of 1.

    Parameters:
        dataset (xarray.Dataset): Input dataset containing the variable.
        variable (str): Name of the variable to compute anomalies for.
        (old ) window (int): Size of the moving window for the second climatology.
        date1 (str): Start date of the reference period (e.g., '1971-01-01').
        date2 (str): End date of the reference period (e.g., '2000-12-31').

    Returns:
        standardized_raw (xarray.DataArray): Standardized anomalies using raw climatology.
        standardized_window (xarray.DataArray): Standardized anomalies using window climatology.
    """
    # Select the reference period
    ds = dataset.sel(time=slice(date1, date2))
    
    # Compute raw climatology
    climatology = ds[variable].groupby('time.dayofyear').mean('time')

    standard_deviation = ds[variable].groupby('time.dayofyear').std('time')
    
    #loess fit for the climatology -----------
    loess_climatology = xr.apply_ufunc(
        loess.loess_ts, 
        climatology, 
        input_core_dims=[["dayofyear"]], 
        output_core_dims=[["dayofyear"]],  
        vectorize=True, 
        dask="parallelized", 
        kwargs={"na_rm": True, "window": 30, "degree": 1}  
    )

    loess_standard_deviation = xr.apply_ufunc(
        loess.loess_ts, 
        standard_deviation,  
        input_core_dims=[["dayofyear"]],  
        output_core_dims=[["dayofyear"]],  
        vectorize=True,  
        dask="parallelized", 
        kwargs={"na_rm": True, "window": 30, "degree": 1} 
    )
    

    # ---------------------------------------

    # Compute anomalies
    anomalies = dataset[variable].groupby('time.dayofyear') - loess_climatology 
  
    # Standardize anomalies
    def standardize(data):
        mean = data.mean('time')
        std = data.std('time')
        return (data - mean) / std
    
    standardized_anomalies = anomalies.groupby('time.dayofyear') / loess_standard_deviation
                                        
    return standardized_anomalies


# Gemini ---------------------------------------------------------------------------------------------------------------------------------------------

def compute_standardized_anomalies(dataset, variable, ref_period_start, ref_period_end):
    """
    Computes LOESS-smoothed, standardized anomalies for a given variable.

    This function operates lazily on Dask-backed xarray objects.

    Parameters:
        dataset (xarray.Dataset): Input dataset, chunked along the 'time' dimension.
        variable (str): Name of the variable to compute anomalies for.
        ref_period_start (str): Start year of the reference period.
        ref_period_end (str): End year of the reference period.

    Returns:
        xarray.DataArray: A Dask-backed DataArray containing the standardized anomalies.
                          Computation is not triggered within this function.
    """
    # 1. Select the reference period for calculating statistics
    ref_ds = dataset.sel(time=slice(ref_period_start, ref_period_end))

    # 2. Compute daily climatology and standard deviation over the reference period.
  
    print("Computing daily statistics for reference period...")
    climatology_mean = ref_ds[variable].groupby('time.dayofyear').mean('time').compute()
    climatology_std = ref_ds[variable].groupby('time.dayofyear').std('time').compute()
    print("Daily statistics computed.")

    # 3. Apply LOESS smoothing to the computed daily statistics.
    print("Applying LOESS smoothing...")
    kwargs = {"na_rm": True, "window": 30, "degree": 1}
    
    loess_climatology = xr.apply_ufunc(
        loess.loess_ts,
        climatology_mean,
        input_core_dims=[["dayofyear"]],
        output_core_dims=[["dayofyear"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs
    )

    loess_std = xr.apply_ufunc(
        loess.loess_ts,
        climatology_std,
        input_core_dims=[["dayofyear"]],
        output_core_dims=[["dayofyear"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs
    )
    print("LOESS smoothing complete.")


    # 4. Calculate anomalies and standardize them.
    print("Building final computation graph for standardization...")
    
    # Group the full dataset by dayofyear to align with the climatology
    grouped_data = dataset[variable].groupby('time.dayofyear')
    
    # Build the lazy calculation
    anomalies = grouped_data - loess_climatology
    standardized_anomalies = anomalies / loess_std

    return standardized_anomalies