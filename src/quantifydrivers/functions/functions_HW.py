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
#import script with loess fucntions 
import loess_functions


#Function for only computing the cliamtology 

def Compute_climatology(dataset,variable,window,date1,date2):

    '''climatology: computed by single days 
    clim_window: computed by averaging moving 5 day wondow over time time period
    percentile: nth percentile computed for a moving m-day window. Percentile calculated per day and then averaged over a 5-day windw'''
    ds = dataset.sel(time=slice(date1,date2))
    climatology = ds[variable].groupby('time.dayofyear').mean('time') #day mean

    #loess fit for the climatology -----------

    loess_climatology = xr.apply_ufunc(
        loess_functions.loess_ts, 
        climatology,  # Your DataArray
        input_core_dims=[["dayofyear"]],  # Apply along "dayofyear"
        output_core_dims=[["dayofyear"]],  # Output keeps same dimension
        vectorize=True,  # Ensures it works element-wise
        dask="parallelized",  # Enable parallelization for large datasets
        kwargs={"na_rm": True, "window": 30, "degree": 1}  # Extra arguments for loess_ts
    )

    # ---------------------------------------

    #window series and climatology of window series 

    window_series = dataset[variable].rolling(time=window,center=True,min_periods=1).mean(skipna=True) #compute window means #window series
    clim_window = window_series.groupby('time.dayofyear').mean('time') #climatology of windows #climatology of window series

    return climatology,clim_window, window_series, loess_climatology


#Function for computing climatology and smothed percentile using a m-day window. 
#The ouptu percentile-n is given so that also the raw computed percentile can be used and, for example, a LOESS fit smoothing can be applied to it. 

def Compute_window_percentile_reference_period(dataset,variable,quantile_value,window,date1,date2):
    '''climatology: computed by single days 
    clim_window: computed by averaging moving 5 day wondow over time time period
    percentile: nth percentile computed for a moving m-day window. Percentile calculated per day and then averaged over a 5-day windw'''
    ds = dataset.sel(time=slice(date1,date2))
    climatology = ds[variable].groupby('time.dayofyear').mean('time') #day mean

    #loess fit for the climatology -----------

    loess_climatology = xr.apply_ufunc(
        loess_functions.loess_ts, 
        climatology,  # Your DataArray
        input_core_dims=[["dayofyear"]],  # Apply along "dayofyear"
        output_core_dims=[["dayofyear"]],  # Output keeps same dimension
        vectorize=True,  # Ensures it works element-wise
        dask="parallelized",  # Enable parallelization for large datasets
        kwargs={"na_rm": True, "window": 30, "degree": 1}  # Extra arguments for loess_ts
    )

    # ---------------------------------------


    #percentile computation 
    ds_rolling = ds[variable].rolling(time=5,center=True).construct("window")
    ds_rolling_grouped = ds_rolling.groupby('time.dayofyear')
    percentile = ds_rolling_grouped.quantile(quantile_value,dim=('time','window'))

    #window series and climatology of window series 

    window_series = dataset[variable].rolling(time=window,center=True,min_periods=1).mean(skipna=True) #compute window means #window series
    clim_window = window_series.groupby('time.dayofyear').mean('time') #climatology of windows #climatology of window series

    return climatology,percentile,clim_window, window_series, loess_climatology


#function for standarized anomalies. Two options, raw or window smoothing

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

    # climatology = climatology.compute()
    
    #loess fit for the climatology -----------
    loess_climatology = xr.apply_ufunc(
        loess_functions.loess_ts, 
        climatology,  # Your DataArray
        input_core_dims=[["dayofyear"]],  # Apply along "dayofyear"
        output_core_dims=[["dayofyear"]],  # Output keeps same dimension
        vectorize=True,  # Ensures it works element-wise
        dask="parallelized",  # Enable parallelization for large datasets
        kwargs={"na_rm": True, "window": 30, "degree": 1}  # Extra arguments for loess_ts
    )

    loess_standard_deviation = xr.apply_ufunc(
        loess_functions.loess_ts, 
        standard_deviation,  # Your DataArray
        input_core_dims=[["dayofyear"]],  # Apply along "dayofyear"
        output_core_dims=[["dayofyear"]],  # Output keeps same dimension
        vectorize=True,  # Ensures it works element-wise
        dask="parallelized",  # Enable parallelization for large datasets
        kwargs={"na_rm": True, "window": 30, "degree": 1}  # Extra arguments for loess_ts
    )
    

    # ---------------------------------------

    
    # Compute anomalies
    anomalies = dataset[variable].groupby('time.dayofyear') - loess_climatology #remove loess climatology 
  
    # Standardize anomalies
    def standardize(data):
        mean = data.mean('time')
        std = data.std('time')
        return (data - mean) / std
    
    standardized_anomalies = anomalies.groupby('time.dayofyear') / loess_standard_deviation

    #standardized_anomalies = standardize(anomalies)
                                        
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
    #    We use .compute() here because the resulting arrays (365 days x lat x lon)
    #    are small enough to fit in memory, which simplifies the LOESS smoothing step.
    print("Computing daily statistics for reference period...")
    climatology_mean = ref_ds[variable].groupby('time.dayofyear').mean('time').compute()
    climatology_std = ref_ds[variable].groupby('time.dayofyear').std('time').compute()
    print("Daily statistics computed.")

    # 3. Apply LOESS smoothing to the computed daily statistics.
    #    This smooths out noise in the daily mean and std dev.
    print("Applying LOESS smoothing...")
    kwargs = {"na_rm": True, "window": 30, "degree": 1}
    
    loess_climatology = xr.apply_ufunc(
        loess_functions.loess_ts,
        climatology_mean,
        input_core_dims=[["dayofyear"]],
        output_core_dims=[["dayofyear"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs
    )

    loess_std = xr.apply_ufunc(
        loess_functions.loess_ts,
        climatology_std,
        input_core_dims=[["dayofyear"]],
        output_core_dims=[["dayofyear"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs
    )
    print("LOESS smoothing complete.")


    # 4. Calculate anomalies and standardize them.
    #    Because `dataset` is a Dask array, these operations are LAZY.
    #    They build a task graph instead of computing results in memory.
    #    Xarray automatically handles broadcasting the smaller `loess` arrays
    #    over the large, chunked `dataset` array.
    print("Building final computation graph for standardization...")
    
    # Group the full dataset by dayofyear to align with the climatology
    grouped_data = dataset[variable].groupby('time.dayofyear')
    
    # Build the lazy calculation
    anomalies = grouped_data - loess_climatology
    standardized_anomalies = anomalies / loess_std

    return standardized_anomalies

    
    


# --------------------- HW detection ------------------------------------------------------------------------------------------------------------------

from scipy.ndimage import label

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
        
    time = dataset.time # Time variable from our xarray
    DateTime_time = np.array([pd.to_datetime(ts.item()) for ts in time.values]) # Convert to datetime format
    
    variable_data = dataset[variable] #T max values 
    #threshold_exceed = np.zeros_like(tasmax, dtype=bool)   # Array to store when the threshold is exceeded
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
        #if (labeled == feature).sum() >= 3:  # Require at least 3 consecutive days in the contiguous region exceeding the 90th percentile. Count TRUE values with .sum()
        HW_mask[:] |= (labeled == feature)  # Assign TRUE values in the HW_mask array if the feature has a minimum duration of 3 days.
        
    #------------------------------------ Heatwave intensity -----------------------------------------------------------------------------------------
    # np.where locates points where HW_mask is TRUE, indicating where the 90th percentile is exceeded.
    # It returns the difference between the t2m value and the 90th percentile (intensity) if HW_mask is TRUE, and 0 otherwise.
    # We check how much the temperature exceeds the 90th percentile.
    
    HW_intensity = np.where(HW_mask, exceedance, 0) # Store values


    # --------------- Dates of HW events -------------------------------------------------------------------------------------------------------------


    # Configuration of the time variable to facilitate data analysis. Basically, we index the dates.

    # ------------- Complete period ---------------------------------

    #Configuration of time variable 
    
    time = dataset.time # Time variable from our xarray
    DateTime_time = np.array([pd.to_datetime(ts.item()) for ts in time.values]) # Convert to datetime format
    #Write heatwave event dates 

    

    # Find the start and end indices of consecutive True values in HW_mask
    hw_changes = np.diff(HW_mask.astype(int))  # Detect changes in HW_mask
    hw_starts = np.where(hw_changes == 1)[0] + 1  # Indices where True starts
    hw_ends = np.where(hw_changes == -1)[0] + 1   # Indices where True ends, event  last day is included in the HW
    
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

    #return matrices 
    #returns DateTime to use in the metrics functions 
    return HW_mask,HW_intensity,DateTime_time

#To do: Create function that computes the HW metrics for a given HW_mask array. Save metrics in a csv file for each of the studied locations.

#def Compute_HW_metrics()


def Compute_metrics(HW_mask,HW_intensity,DateTime_time):

    """This function computes various metrics given an array with masked data of extreme events. 
    heat_wave_mask (bool): numpy array with true values in space and time for detected extreme event 
    
    """

    #--------------------- event == heatwave ----------------------------------------------------------

    
    # Label the mask array. Heatwaves that we have marked as valid because they last more than 3 days. 
    # Detect the number of events in the mask array.
    labeled, num_features = label(HW_mask) # HW_mask marks all values with True and False depending on whether the point (time, lat, lon) belongs to a detected heatwave.
    #This labeled creates events of various True values together in time. 
    #marks every day with True value with a unique integer. 
    
    # Arrays to store metrics
    duration = np.zeros((HW_mask.shape[0])) # Array to store the duration of events
    #frequency = np.zeros((HW_mask.shape[1], HW_mask.shape[2])) # Contains the number of events for each point (lat, lon), the frequency
    #max_intensity = np.zeros((HW_mask.shape[1], HW_mask.shape[2])) # Maximum intensity of an event at each grid point
    #mean_intensity = np.zeros((HW_mask.shape[1], HW_mask.shape[2])) # Average intensity of events at the grid point
    cumulative_intensity = np.zeros((HW_mask.shape[0])) # Cumulative intensity of different events
    
    # Loop through all grid points
    #for lat in range(HW_mask.shape[1]):  # Loop over latitude
     #   for lon in range(HW_mask.shape[2]):  # Loop over longitude
    # Time series for the grid point 
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
        # If the event appears in 3 time points at the grid point, they are marked as 1, 2, 3. Therefore, HW_0_1the event duration is 3 days.
        cumulative_intensity[event_mask] = event_cumulative  # Cumulative intensity for this event
        max_intensity = event_intensity.max() # Update maximum intensity for this grid point
        mean_intensity = event_intensity.mean()  # Add to mean intensity (later divided by the number of events)


        # Frequencies for the individual periods 
        event_dates = DateTime_time[event_mask]  # Extract corresponding dates
    
        # Count event frequency for each period. 
        # We are adding 1 because the frequency is for events of HW, not all days surpassing the 90th percentile. 
        # This is to track how many HW events we have. 
        # The percentage computed below is for all the days surpassing the 90th percentile, and should give 10% of extreme days and 90% of normal days aprox.
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
    print(f"All days 1950-2000 = {np.sum(mask_1950_2000)}")
    period_counts["1971-2000"]["All_days"] = np.sum(mask_1971_2000)  # Normal days in 1971-2000
    period_counts["2001-2024"]["All_days"] = np.sum(mask_2001_2024)  # Normal days in 2001-2024

        
    # Calculate percentages for each period using period_counts dictionary
    period_percentages = {
        period: {
            "HW_days_percent": (period_counts[period]["HW_days"] / (period_counts[period]["All_days"])) * 100,
        }
        for period in period_counts
    }

    print(period_counts)
    print(period_percentages)

    
    # Divide mean intensity by the number of events 
    if len(events) > 0:
        mean_intensity /= len(events)
    
    # Array with the maximum temporal duration for each grid point
    max_duration = np.max(duration, axis=0)

    #list with the individual frequencies for output 
    frequencies_selected_periods = {'1950-2000':frequency_1950_2000,'1971-2000':frequency_1971_2000,'2001-2024':frequency_2001_2024}

    print(frequencies_selected_periods)

    return duration,frequency,frequencies_selected_periods,max_intensity,mean_intensity,cumulative_intensity,period_percentages,period_counts

    


#------------------------------------------------------------------------------------------------------------------------------------

   
