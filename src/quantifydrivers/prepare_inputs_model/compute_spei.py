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


# --- 1. General Configuration -----------------------------------------------

# Directory where your 'combined_climate_data' files are stored
data_directory = "/path/to/your/data/input/"
output_directory = "/path/to/your/data/output/"

# List of sites to process
sites = ["cordoba","hannover", "stockholm", "lyon", "belgrado", "marrakech"]

site_latitudes = {
    "cordoba": 37.88,
    "hannover": 52.37,
    "stockholm": 59.33,
    "lyon": 45.76,
    "belgrado": 44.78,
    "marrakech": 31.63
}

# ===========================================================================
# --- 2. SPEI Calculation Parameters ---

method = "hg"  # PET calculation method: 'thorn' or 'hg'
scale = 90     # SPEI timescale in days
distribution = 'gamma'  
daily_or_monthly = 'daily'  # Indicates if the data is daily or monthly
ref_years = np.arange(1951, 2000)  # Reference period
using_era5land = True  # Set to True if using ERA5-Land data 
# ===========================================================================


# --- 3. Main Processing Loop ---

print(f" Starting SPEI calculation for {len(sites)} sites...")
print(f"   Method: {method.upper()}, Timescale: {scale}-day")

for site in sites:
    print(f"\n--- Processing: {site.capitalize()} ---")
    
    try:
        # Construct the input filename for the current site
        input_filename = f"combined_era5land_land_data_{site}.nc"
        input_path = os.path.join(data_directory, input_filename)
        
        # Load the combined dataset for the site
        climate_data = xr.open_dataset(input_path).sel(time=slice('1951','2024'))

        # Ensure proper units for variables
        for var in ["tas", "tasmin", "tasmax", "pr"]:
            if "units" not in climate_data[var].attrs or not isinstance(climate_data[var].attrs["units"], str):
                print(f"  -> WARNING: {var} has missing or invalid units. Setting manually.")
                if var == "pr":
                    climate_data[var].attrs["units"] = "mm/day"
                elif var in ["tas", "tasmin", "tasmax"]:
                    climate_data[var].attrs["units"] = "K"
                
        print(f"  -> Successfully loaded {input_filename}")

        site_lat = xr.DataArray(site_latitudes[site], name="lat", attrs={"units": "degrees_north"})

        # Define the unique name for the new SPEI variable
        indx_name = f"spei_{method}_{scale}"

        # --- Calculate Potential Evapotranspiration (PET) ---
        print(f"  -> Calculating Potential Evapotranspiration (PET) using '{method}' method...")

        # IF statement to ensure no repeated dates are present due two differences in the hours of the time series
        if using_era5land:
            # Check for duplicate dates
            daily_timestamps = climate_data['time'].dt.floor('D')
            duplicate_series = daily_timestamps.to_series().duplicated(keep='first')
            duplicate_mask = xr.DataArray(duplicate_series, coords=daily_timestamps.coords, dims=daily_timestamps.dims)

            if duplicate_mask.any():
                print(f"  -> WARNING: Found duplicate daily timestamps for {site}. Removing duplicates.")
                duplicate_dates = daily_timestamps[duplicate_mask].values
                print(f"  -> Duplicate dates found: {np.unique(duplicate_dates)}")
                climate_data = climate_data.sel(time=~duplicate_mask)
            else:
                print(f"  -> No duplicate daily timestamps found for {site}.")

            climate_data = climate_data.resample(time='D').mean()

        if method == "thorn":
            climate_data['pet'] = xclim.potential_evapotranspiration(
                tas=climate_data.tas, 
                method="thornthwaite",
                lat=site_lat
            )
        elif method == "hg": 
            climate_data['pet'] = xclim.potential_evapotranspiration(
                tas=climate_data.tas,
                tasmin=climate_data.tasmin,
                tasmax=climate_data.tasmax,
                method="HG85",
                lat=site_lat
            )
            print('Attributes of PET variable before conversion:')
            print(climate_data['pet'].attrs)
            if climate_data['pet'].attrs.get('units') in ['kg m-2 s-1', 'kg/m^2/s']:
                climate_data['pet'] = climate_data['pet'] * 86400
                climate_data['pet'].attrs['units'] = 'mm/day'

            if climate_data['pr'].attrs.get('units') in ['kg m-2 s-1', 'kg/m^2/s']:
                climate_data['pr'] = climate_data['pr'] * 86400
                climate_data['pr'].attrs['units'] = 'mm/day'
        else:
            print(f"  -> ERROR: Method '{method}' not recognized! Skipping site.")
            continue

        print('Attributes of PET variable after conversion:')
        print(climate_data['pet'].attrs)

        # --- Calculate Climatic Water Balance (WB) ---
        print("  -> Calculating climatic water balance...")
        wb = xclim.water_bu_
