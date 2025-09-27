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


# --- 1. General Configuration ---

# Directory where your 'combined_climate_data' files are stored
data_directory = "/path/to/your/data/input/"
output_directory = "/path/to/your/data/output/"

# List of sites to process
sites = ["cordoba", "hannover", "stockholm", "lyon", "belgrado"]

# --- 2. SPI Calculation Parameters ---

scale = 90 # Define the SPI timescale in days
ref_years = np.arange(1951, 2000) # Define the reference period for calibrating the index
distribution = 'gamma'  # Distribution for SPEI calculation

# --- 3. Main Processing Loop ---

print(f"Starting SPI calculation for {len(sites)} sites...")
print(f"   Timescale: {scale}-day")

# --- 2. Loop through each site ---

for site in sites:
    print(f"\n--- Processing: {site.capitalize()} ---")
    
    try:
        # Construct the input filename for the current site
        input_filename = f"combined_era5land_land_data_{site}.nc"
        input_path = os.path.join(data_directory, input_filename)
        
        # Load the combined dataset for the site
        climate_data = xr.open_dataset(input_path)
        climate_data = climate_data.drop_vars("height")
        print(f"  -> Successfully loaded {input_filename}")

        # Define the unique name for the new SPI variable
        indx_name = f"spi_{scale}"

        # --- Calculate SPI ---
        print(f"  -> Calculating {scale}-day SPI...")

        # 1. Resample the daily precipitation to daily sums
        pr = climate_data.pr
        pr.attrs['units'] = 'mm/day'  # Ensure correct units for SPI calculation
        
        # 2. Select the calibration data
        cal_strt, cal_end = ref_years[0], ref_years[-1]

        # 3. Calculate the SPI
        monthly_spi = xclim.standardized_precipitation_index(
            pr,
            freq='D',
            window=scale,
            dist=distribution,  
            method='ML',
            cal_start=f"{cal_strt}-01-01",
            cal_end=f"{cal_end}-12-31"
        )

        # 4. Align time
        climate_data['time'] = climate_data.indexes['time'].normalize()
        monthly_spi = monthly_spi.reindex(time=climate_data.time)
        climate_data[indx_name] = monthly_spi
              
        # --- Filter and Save ---
        print(f"  -> Filtering dataset to keep 'tasmax' and '{indx_name}'...")
        output_ds = climate_data[['tasmax', indx_name]]
        
        output_filename = f"era5land_{distribution}_spi_tasmax_{site}_{scale}_daily.nc"
        output_path = os.path.join(output_directory, output_filename)
        
        # Save the filtered dataset
        output_ds.to_netcdf(output_path)
        print(f"✅ Success! Filtered data saved to: {output_path}")

    except FileNotFoundError:
        print(f"  -> ERROR: File not found at {input_path}. Skipping this site.")
    except Exception as e:
        print(f"  -> An unexpected error occurred for site '{site}': {e}")

print("\nAll sites processed successfully!")
