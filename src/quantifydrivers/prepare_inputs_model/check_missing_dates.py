
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

# ************************************************
# Code to check missing dates between two datasets
# ************************************************

site = "cordoba"          # Replace with your site name
swvl = "swvl1"            # Soil moisture variable (swvl1, swvl2, swvl3)
spei_scale = 30           # SPI scale (e.g., 30, 60, 90)

# Load soil moisture data
swvl_da = xr.open_dataset(
    f"/path/to/your/data/era5_land/variables_era5land_data_{site}_1950_2024.nc"
).sel(time=slice('1952','2024'))[swvl]

# Load SPEI/SPI data
spei_da = xr.open_dataset(
    f"/path/to/your/data/era5_land/spei_computations/era5land_gamma_spi{spei_scale}_tasmax_{site}_daily.nc"
)[f'spi_{spei_scale}'].sel(time=slice('1952','2024'))

# Extract time values as pandas DatetimeIndex
time_swvl = swvl_da.time.to_index().normalize()
time_spei = spei_da.time.to_index().normalize()

# Now compare
missing_in_spei = time_swvl.difference(time_spei)
missing_in_swvl = time_spei.difference(time_swvl)

print("Dates in swvl_era5_land but missing in spei_era5land:")
print(missing_in_spei)

print("\nDates in spei_era5land but missing in swvl_era5_land:")
print(missing_in_swvl)
