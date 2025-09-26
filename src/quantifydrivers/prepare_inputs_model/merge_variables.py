
# ==========================================================================================
# IMPORT NEEDED LIBRARIES
# ==========================================================================================

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr 
from scipy.stats import linregress
from datetime import datetime, timedelta
import pandas as pd
import os
import argparse
import xclim

# ==========================================================================================

# ***************************************************************************************************************************
# Code to merge multiple NetCDF files into a single file per site. This output file is then used for the SPEI/SPI computation. 
# ***************************************************************************************************************************

# --- 1. Configuration ---

# The folder containing your NetCDF files
input_directory = "/path/to/your/data"

# The sites you want to process
sites = ["cordoba","hannover", "stockholm", "lyon", "belgrado"]

# Target variable names in the final file
variables = ["prlr", "tas", "tasmin", "tasmax"]
# Variable names in the original source files
variables_file = ["prlr", "tas", "tasmin", "TX"]

# --- 2. Main Processing Loop ---
print("Starting to merge NetCDF files with strictly enforced string units...")

for site in sites:
    print(f"\nProcessing site: {site.capitalize()}")
    
    datasets_to_merge = []
    
    # Loop through each variable to load, process, and assign units
    for var_name, var_name_file in zip(variables, variables_file):
        
        if var_name == "tasmax":
            file_path = f"/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/variables_era5land_data_{site}_1950_2024.nc"
            ds = xr.open_dataset(file_path)
        else:
            file_path = os.path.join(input_directory, f"era5land_{var_name_file}_data_{site}_1950_2024.nc")
            # Open the dataset      
            ds = xr.open_dataset(file_path)
        
        # --- Handle each variable to ensure correct name and string units ---
        
        if var_name == "prlr":
            print(f"  - Converting 'prlr' to daily 'pr'...")
            conversion_factor = 86400000
            ds['prlr'] = ds['prlr'] * conversion_factor
            ds = ds.rename({'prlr': 'pr'})
            # Explicitly set units as a string
            ds['pr'].attrs['units'] = str('mm/day') 
            ds['pr'].attrs['long_name'] = 'Daily Total Precipitation'

            ds_to_add = ds[['pr']]


        elif var_name == "tasmax":
            print(f"  - Renaming 'TX' to 'tasmax' and setting units...")
            # Explicitly set units as a string
            ds['tasmax'].attrs['units'] = str('K')
            ds['tasmax'].attrs['long_name'] = 'Daily Maximum Temperature'

            ds_to_add = ds[['tasmax']]

        else: # This handles 'tas' and 'tasmin'
            print(f"  - Ensuring units for '{var_name}' are set...")
            # Explicitly set units as a string
            ds[var_name].attrs['units'] = str('K')
            if var_name == 'tas':
                ds['tas'].attrs['long_name'] = 'Daily Mean Temperature'
            elif var_name == 'tasmin':
                ds['tasmin'].attrs['long_name'] = 'Daily Minimum Temperature'

            ds_to_add = ds[[var_name]]

        datasets_to_merge.append(ds)

    # Merge all the datasets for the site into one
    combined_ds = xr.merge(datasets_to_merge,compat='override').sel(time=slice('1951','2024'))
    combined_ds = combined_ds.drop_vars(['swvl1','swvl2','swvl3'])
    
    output_filename = f"combined_era5land_land_data_{site}.nc"
    output_path = os.path.join(input_directory, output_filename)
    
    # Save the merged dataset, overwriting old files
    combined_ds.to_netcdf(output_path)
    
    print(f"Combined file with guaranteed string units saved to: {output_path}")

print("\nAll sites processed successfully!")
