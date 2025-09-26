
# =========================================================
# IMPORT NEEDED LIBRARIES
# =========================================================

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
import re
import ast
import xesmf as xe#for regridding 
from matplotlib.backends.backend_pdf import PdfPages
from ../functions import data_preprocess as dp

# =========================================================

# Sites to process
sites = ['cordoba', 'hannover', 'stockholm', 'lyon', 'belgrado', 'marrakech']

# Loop through each site

for site in sites:

    path = f'/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/variables_era5land_data_{site}_1950_2024.nc'
    ds = xr.open_dataset(path).sel(time=slice('1950', '2024'))
    
    # Create a copy for detrended data
    ds_detrended = xr.Dataset()
    
    # Loop through each variable and detrend
    for var in ds.data_vars:
        da = ds[var]

        time_index = xr.DataArray(
            np.arange(len(da.time)),
            dims=['time_index'],
            coords={'time_index': np.arange(len(da.time))},
            name='time_index'
        )

        
        da_with_index = xr.DataArray(
            da.values,
            dims=['time_index', *da.dims[1:]],
            coords={
                'time_index': time_index,
                **{dim: da[dim] for dim in da.dims[1:]}
            }
        )
        
        # 4. Now perform the fit using time_index dimension
        fit = da_with_index.polyfit(dim='time_index', deg=1, skipna=True)
        
        # 5. Evaluate trend
        trend_line = xr.polyval(da_with_index['time_index'], fit['polyfit_coefficients'])
        
        # 6. Detrend and add to original dataset (swap back to time dimension)
        ds_detrended[f'{var}_detrended'] = (da.dims, (da.values - trend_line.values))
        
        # --- Verification ---
        slope = fit['polyfit_coefficients'].sel(degree=1)
        print(f"Detrending slope: {slope.item()}")
    
    # Save detrended dataset to NetCDF
    output_path = f'/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/detrended_variables_era5land_data_{site}_1950_2024.nc'
    ds_detrended.to_netcdf(output_path)