

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
from scipy.ndimage import label
from . import loess

# ========================================================================================================================

# =========================================================================================================================
# This script provides a set of functions to compute climate-related indices and detect extreme heatwave events.
# 
# Main functionalities:
#   - Detect heatwave events using percentile-based thresholds.
#   - Compute heatwave metrics such as frequency, duration, intensity, and cumulative measures.
#
# Dependencies: numpy, xarray, pandas, matplotlib, cartopy, scipy, netCDF4, and custom loess.
# =========================================================================================================================

# --------------------- HW detection ------------------------------------------------------------------------------------------------------------------


def detect_HW(dataset,variable,percentile,duration,site):

    '''Function for detecting heat_wave vevents in a dataset with data fro max temeprature.

    dataset: data with Tmax values
    Percentile: nth percentile for the HW detection
    Duration: Minimum duration required to classify an event as HW
    
    HW_mask: temporal and spatial array that gives true when a space-time point corresponds to a heatwave event.
    HW_intensity: saves intensity of the different HW events. '''

    # Configuration of the time variable to facilitate data analysis. Basically, we index the dates.

    # ------------- Complete period ---------------------------------

    #remove 29th of febraury
    if 366 in percentile.dayofyear:
        dataset = dataset.sel(time=~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29)))

    # --------------------------------------------------------------
        
    time = dataset.time 
    DateTime_time = np.array([pd.to_datetime(ts.item()) for ts in time.values]) # Convert to datetime format
    
    variable_data = dataset[variable] #T max values 
    HW_mask = np.zeros_like(variable_data, dtype=bool)   
    variable_data_diff = np.zeros_like(variable_data) # To store the difference between temperatures and the 90th percentile, used later for intensity calculations.

    #------------Calculate exceedance and events --------------------------------------
        
    # Broadcast percentile values to all years using groupby
    exceedance = variable_data.groupby("time.dayofyear") - percentile
    threshold_exceed = exceedance > 0 #True if values surpass the nth percentile
    
    HW_mask = np.zeros_like(exceedance.values, dtype=bool) # Array to store when a heatwave (HW) occurs (previously defined). TRUE if it is, FALSE otherwise

    # Label consecutive events where the nth percentile is exceeded
    labeled, num_features = label(threshold_exceed) 

    for feature in range(1, num_features + 1): # Iterate over events in the grid point
        # ****** Optional if statmeent to set a minimum duration for the event to be considered a heatwave. **************************************
        # I our case we consider extreme events as individual days and do not set a minimum duration.
        #if (labeled == feature).sum() >= 3:  # Require at least 3 consecutive days in the contiguous region exceeding the 90th percentile. Count TRUE values with .sum()
        HW_mask[:] |= (labeled == feature)  # Assign TRUE values in the HW_mask array if the feature has a minimum duration of 3 days.
        
    #------------------------------------ Heatwave intensity -----------------------------------------------------------------------------------------

    # It returns the difference between the t2m value and the 90th percentile (intensity) if HW_mask is TRUE, and 0 otherwise.
    # We check how much the temperature exceeds the 90th percentile.
    
    HW_intensity = np.where(HW_mask, exceedance, 0) # Store values

    # --------------- Dates of HW events -------------------------------------------------------------------------------------------------------------

    # Configuration of the time variable to facilitate data analysis. Basically, we index the dates.

    time = dataset.time 
    DateTime_time = np.array([pd.to_datetime(ts.item()) for ts in time.values]) # Convert to datetime format

    # Find the start and end indices of consecutive True values in HW_mask
    hw_changes = np.diff(HW_mask.astype(int))     # Detect changes in HW_mask
    hw_starts = np.where(hw_changes == 1)[0] + 1  # Indices where True starts
    hw_ends = np.where(hw_changes == -1)[0] + 1   # Indices where True ends, last day included
    
    # Handle edge cases where a heatwave starts at the beginning or ends at the end of the array
    if HW_mask[0]:  # If the first value is True
        hw_starts = np.insert(hw_starts, 0, 0)
    if HW_mask[-1]:  # If the last value is True
        hw_ends = np.append(hw_ends, len(HW_mask))
    
    # Calculate durations and extract start dates
    heatwave_events = []
    for start, end in zip(hw_starts[:], hw_ends[:]):
        duration = end - start  # Duration of the heatwave
        start_date = DateTime_time[start]  # First date of the heatwave
        heatwave_events.append((start_date, duration))
    
    # Convert to a DataFrame
    heatwave_df = pd.DataFrame(heatwave_events, columns=['Date', 'Duration'])
    
    # Write to CSV
    heatwave_df.to_csv(f'HW_dates_detected_and_metrics/heatwave_events_1950_2024_{site}.csv', index=False)
    
    print(f"Heatwave events saved to 'heatwave_events_1950_2024_{site}.csv'")

    #Return matrices 
    #Return DateTime to use in the metrics functions 
    return HW_mask,HW_intensity,DateTime_time


# --------------------- Compute metrics of detected events --------------------------------------------------------------------------------------------


def Compute_metrics(HW_mask,HW_intensity,DateTime_time):

    """This function computes various metrics given an array with masked data of extreme events. 
    heat_wave_mask (bool): numpy array with true values in space and time for detected extreme event 
    
    """

    #--------------------- event == heatwave ----------------------------------------------------------

    
    # Label the mask array. Heatwaves that we have marked as valid.
    # Detect the number of events in the mask array.
    labeled, num_features = label(HW_mask) # HW_mask marks all values with True and False depending on whether the point (time, lat, lon) belongs to a detected heatwave.
    
    # Arrays to store metrics
    duration = np.zeros((HW_mask.shape[0])) # Array to store the duration of events
    cumulative_intensity = np.zeros((HW_mask.shape[0])) # Cumulative intensity of different events
  
    gridpoint_labels = labeled[:]
    
    # Extract labels (events) of different features (consecutive True values along the time dimension)
    events = np.unique(gridpoint_labels[gridpoint_labels > 0]) # Integer labels for all heatwave events in HW_mask

    # Get frequencies of events for selected periods and for global period --------------------------------

    # Define time period masks with pd.Timestamp (replace np.datetime64)
    mask_1950_2000 = (DateTime_time >= pd.Timestamp("1950-01-01")) & (DateTime_time <= pd.Timestamp("2000-12-31"))
    mask_1971_2000 = (DateTime_time >= pd.Timestamp("1971-01-01")) & (DateTime_time <= pd.Timestamp("2000-12-31"))
    mask_2001_2024 = (DateTime_time >= pd.Timestamp("2001-01-01")) & (DateTime_time <= pd.Timestamp("2024-12-31"))
    
    # Initialize frequency counters for the selected periods 
    frequency_1950_2000 = 0
    frequency_1971_2000 = 0
    frequency_2001_2024 = 0
    
    #global frequency    
    frequency = len(events)  # Number of events at this grid point

    # Analisis percentage of normal days in the diferent periods
    # Initialize dictionary to store percentage results
    period_counts = {
        "1950-2000": {"HW_days": 0, "All_days": 0},
        "1971-2000": {"HW_days": 0, "All_days": 0},
        "2001-2024": {"HW_days": 0, "All_days": 0}
    }
    
    # Cumulative (integrated) intensity for the grid point being analyzed
    cumulative_intensity_event = np.zeros(HW_mask.shape[0])  
    #Loop over events and get the different metrics 
    for event in events: # Loop through events in the grid point
        # Mask for the event being analyzed 
        event_mask = gridpoint_labels == event # Temporal mask of the event in the time series of the grid point
    
        # Cumulative intensity of the event
        progressive_counter = np.arange(1, np.sum(event_mask) + 1)  # Count event duration along the time dimension for this point (lat, lon)
        event_intensity = HW_intensity[event_mask]  # Intensities of the event 
        event_cumulative = np.cumsum(event_intensity)  # Cumulative intensity of the event 
    
        # Update metric arrays 
        duration[event_mask] = progressive_counter  # Event duration at the grid point, adding 1 for each day the event occurs
        cumulative_intensity[event_mask] = event_cumulative  # Cumulative intensity for this event
        max_intensity = event_intensity.max() # Update maximum intensity for this grid point
        mean_intensity = event_intensity.mean()  # Add to mean intensity (later divided by the number of events)


        # Frequencies for the individual periods 
        event_dates = DateTime_time[event_mask]  # Extract corresponding dates
    
        # Count event frequency for each period. 
        if np.any(mask_1950_2000 & event_mask):
            frequency_1950_2000 += 1
        if np.any(mask_1971_2000 & event_mask):
            frequency_1971_2000 += 1
        if np.any(mask_2001_2024 & event_mask):
            frequency_2001_2024 += 1

        #Extreme and normal days counting
        # Count heatwave (True) and normal (False) days for each period
        period_counts["1950-2000"]["HW_days"] += np.sum(event_mask & mask_1950_2000)  # Heatwave days in 1950-2000
        period_counts["1971-2000"]["HW_days"] += np.sum(event_mask & mask_1971_2000)  # Heatwave days in 1971-2000
        period_counts["2001-2024"]["HW_days"] += np.sum(event_mask & mask_2001_2024)  # Heatwave days in 2001-2024
        

    period_counts["1950-2000"]["All_days"] = np.sum(mask_1950_2000)  # Normal days in 1950-2000
    period_counts["1971-2000"]["All_days"] = np.sum(mask_1971_2000)  # Normal days in 1971-2000
    period_counts["2001-2024"]["All_days"] = np.sum(mask_2001_2024)  # Normal days in 2001-2024

        
    # Calculate percentages for each period using period_counts dictionary
    period_percentages = {
        period: {
            "HW_days_percent": (period_counts[period]["HW_days"] / (period_counts[period]["All_days"])) * 100,
        }
        for period in period_counts
    }

    
    # Divide mean intensity by the number of events 
    if len(events) > 0:
        mean_intensity /= len(events)
    
    # Array with the maximum temporal duration for each grid point
    max_duration = np.max(duration, axis=0)

    #List with the individual frequencies for output 
    frequencies_selected_periods = {'1950-2000':frequency_1950_2000,'1971-2000':frequency_1971_2000,'2001-2024':frequency_2001_2024}

    print(frequencies_selected_periods)

    return duration,frequency,frequencies_selected_periods,max_intensity,mean_intensity,cumulative_intensity,period_percentages,period_counts


#------------------------------------------------------------------------------------------------------------------------------------