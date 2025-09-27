

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
#   - Regridding:
#       * regridded_dataset: Regrid data to a 1°x1° grid using bilinear interpolation (via xesmf).
#
# Typical use cases:
#   - Preparing ERA5 or climate reanalysis datasets for training artificial neural networks (ANNs).
#   - Standardizing spatial resolution across datasets.
#
# Dependencies: numpy, xarray, pandas, matplotlib, cartopy, netCDF4, scipy, xesmf, os, datetime
# =========================================================================================


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