
# =====================================================================================================================
# IMPORT NEEDED PACKAGES
# =====================================================================================================================

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

from tools import lagged_data, loess

# =====================================================================================================================

# Code to reproduce figures from Mahlstein et al., 2015. Includes raw and smoothed time series of the 90th percentile and climatology of daily maximum temperature (tasmax) 
# for different the different locations.

sites = ['cordoba','hannover','stockholm','lyon','belgrado','marrakech']
sites_labels = ['Córdoba','Hannover','Stockholm','Lyon','Belgrado','Marrakech']

#Defined studied variable 
variable = 'tasmax'

for site,site_label in zip(sites,sites_labels): 

    #path = f'/gpfs/scratch/bsc32/bsc167965/tfm_data/era5_land/variables_era5land_data_{site}_1950_2024.nc'
    #ds_swvl = xr.open_dataset(path)  
    path =  f'/your/path/to/data/data_{site}.nc'
    ds = xr.open_dataset(path)

    # Ensure time is in datetime format
    ds['time'] = pd.to_datetime(ds['time'])

    # Compute daily min and max temperatures across all years
    daily_min = ds[variable].groupby("time.dayofyear").min("time")
    daily_max = ds[variable].groupby("time.dayofyear").max("time")
    # Get day-of-year for x-axis
    days = daily_min.dayofyear

    #extract climatology, window-climatology, percentile
    #Can be computed for a certain reference period or for the full available time period
    climatology,percentile,clim_window,window_series,loess_clim = functions_HW.Compute_window_percentile_reference_period(ds,variable,0.9,5,'1950','2000')

    #apply LOESS to percentile array
    percentile_LOESS = loess.loess_ts(percentile, na_rm=True,  window=40,  degree=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12.6, 10))
    
    # Plot daily tasmax
    ax.plot(climatology['dayofyear'].values, climatology.values, color='blue', alpha=0.5, label='Raw climatology')
    
    # Overlay 90th percentile with same x-axis
    ax.plot(percentile['dayofyear'], percentile, color='orange', linewidth=2, marker='o', markersize=4, label='5-day pull 90th Percentile')
    ax.plot(percentile['dayofyear'], percentile_LOESS, color='black', linewidth=0.25, marker='o', markersize=4, label='LOESS smothed 90th Percentile')
    
    #n-window
    ax.plot(clim_window['dayofyear'], clim_window.values, color='red', linewidth=0.1, marker='o', markersize=4, label='5-day window pull climatology')

    #loess clim 
    ax.plot(loess_clim['dayofyear'], loess_clim.values, color='#4B0082', linewidth=0.01, marker='o', markersize=4, label='LOESS-fit climatology')

    #background with all the series 
    ax.fill_between(days, daily_min, daily_max, color="gray", alpha=0.3, label="Raw series")

    # Add text box with count of extreme events ------------------------------------- 

    file_path = f'HW_dates_detected_and_metrics/heatwave_stats_{site}.txt'
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract values using regex
    freq_selected = re.search(r"Count normal and extreme days per period: (.*)", content).group(1)
    freq_selected = freq_selected.replace("np.int64", "")
    percentages = re.search(r"Percentages normal and extreme days per period: (.*)", content).group(1)
    percentages = re.sub(r"np\.float64\(([\d.eE+-]+)\)", r"\1", percentages)
    
    # Convert string dictionaries to actual Python dictionaries
    freq_selected_dict = ast.literal_eval(freq_selected)
    percentages_dict = ast.literal_eval(percentages)
    
    # Extract required values
    freq_1950_2000 = freq_selected_dict['1950-2000']['HW_days']
    freq_1971_2000 = freq_selected_dict['1971-2000']['HW_days']
    freq_2001_2024 = freq_selected_dict['2001-2024']['HW_days']
    total_frequency_extremes = freq_1950_2000 + freq_2001_2024
    
    percent_1950_2000 = percentages_dict['1950-2000']['HW_days_percent']
    percent_1971_2000 = percentages_dict['1971-2000']['HW_days_percent']
    percent_2001_2024 = percentages_dict['2001-2024']['HW_days_percent']
    
    # Create the text box content
    text_box = (f"Total count extremes: {total_frequency_extremes} \n"
                f"Count extremes 1950-2000: {freq_1950_2000}\n"
                f"% extreme days 1950-2000: {percent_1950_2000:.1f}%\n"  
                f"Count extremes 2001-2024: {freq_2001_2024}\n"
                f"% extreme days 2001-2024: {percent_2001_2024:.1f}%")
    
    ax.text(0.815, 0.15, text_box, transform=ax.transAxes, fontsize=24,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    

    # Add general features of the plot --------------------------------------------
    
    # Labels and title
    ax.set_xlabel('Day of Year',fontsize=24)
    ax.set_ylabel(r'T$_{max}$ [K]',fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_title(f'{site_label}',fontsize=16)
    handles, labels = ax.get_legend_handles_labels()

    # Reorder (example: if you want label2, then label1, then label3)
    order = [2, 3, 1, 4, 0, 5]  # indices in desired order
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='upper left', frameon=False, fontsize=22)

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig(f'figures_mahlstein/mahlstein_figure_{site}.png', dpi=300, bbox_inches='tight')

    plt.show()