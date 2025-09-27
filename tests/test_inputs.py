import pytest
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

from quantifydrivers.functions_inputs import loess_functions
from quantifydrivers.functions_inputs import data_preprocess
from quantifydrivers.functions_inputs import features_labels

# ==============================================================================
# Fixtures for Dummy Data
# ==============================================================================

@pytest.fixture
def dummy_timeseries():
    """Returns a simple 1D numpy array for LOESS testing."""
    ts = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.5 * np.random.rand(100)
    return ts

@pytest.fixture
def dummy_xarray_dataset():
    """
    Returns a multi-variable xarray.Dataset suitable for all input tests.
    Contains 730 days, 1 lat, 1 lon point.

    Variables introduced are tasmax and swvl1
    """
    # Create time dimension for 2 non-leap years (730 days)
    dates = pd.date_range('2001-01-01', periods=730, freq='D')
    
    # lat, lon coordinates
    lat = np.array([40])
    lon = np.array([-60])
    
    # Data Variables (Time, Lat, Lon)
    # 1. tasmax
    data_tasmax = np.random.rand(len(dates), len(lat), len(lon))
    # 2. swvl1 
    data_swvl1 = np.random.rand(len(dates), len(lat), len(lon))
    
    ds = xr.Dataset(
        {
            # Dummy dataset
            'tasmax': (('time', 'lat', 'lon'), data_tasmax),
            'swvl1': (('time', 'lat', 'lon'), data_swvl1) 
        },
        coords={'time': dates, 'lat': lat, 'lon': lon}
    )
    return ds

# ==============================================================================
# Test loess_functions.py
# ==============================================================================

def test_loess_ts_output_shape(dummy_timeseries):
    """Tests if loess_ts executes and returns a numpy array of the correct shape."""
    ts = dummy_timeseries
    window = 30 # rolling-window for smoothinf
    degree = 1 # degree of fitting 
     
    result = loess_functions.loess_ts(ts, na_rm=True, window=window, degree=degree)
    
    # Output is a numpy array
    assert isinstance(result, np.ndarray)
    # Same shape as input 
    assert result.shape == ts.shape
    # Ensure output not too close to the original 
    assert not np.allclose(result, ts, atol=0.1)

# ==============================================================================
# Test data_preprocess.py
# ==============================================================================

def test_create_lagged_features_single_correctness(dummy_xarray_dataset):
    """Tests if the lagged features are created with the correct shift."""
    ds = dummy_xarray_dataset # initialize dummy dataset
    var = 'tasmax'
    lags = [1, 3]
    new_prefix = 'lag_'
    
    # Store the original data array for comparison
    original_data_array = ds[var] 

    # The function returns a new Dataset containing only the lagged variables
    ds_lagged = data_preprocess.create_lagged_features_single(ds, var, lags, new_prefix)
    
    # 1. Check that lagged variables are present
    lag1_var = f'{new_prefix}{var}_anomalies_lag1'
    lag3_var = f'{new_prefix}{var}_anomalies_lag3'
    
    assert lag1_var in ds_lagged
    assert lag3_var in ds_lagged
    
    # 2. Check dimensions
    assert ds_lagged[lag1_var].dims == ds[var].dims
    
    # 3. Check the shift logic for lag=1
    
    original_flat = original_data_array.values.flatten()
    lag1_flat = ds_lagged[lag1_var].values.flatten()

    # First value should be nan for lag1 (first three for lag3)
    assert np.isnan(lag1_flat[0])
    # The rest should match the original shifted by 1
    assert np.allclose(lag1_flat[1:], original_flat[:-1], equal_nan=False)

# ==============================================================================
# Test data_preprocess.py (Multiple Features Lag)
# ==============================================================================

def test_create_lagged_features_multiple_correctness(dummy_xarray_dataset):
    """Tests if multiple lagged features are created correctly with different lags.
    In thistest, we have two laged variables for tasmax and for swvl1"""
    ds = dummy_xarray_dataset
    variables = ['tasmax', 'swvl1']
    lag_dict = {'tasmax': [1, 2], 'swvl1': [5]} # Different lags for each variable
    new_prefix = 'era5_'

    # The function returns a copy of the original dataset with new variables added
    ds_lagged = data_preprocess.create_lagged_features_multiple(ds, variables, lag_dict, new_prefix)

    # 1. Check for expected new variables (NOTE: This function does NOT add '_anomalies')
    expected_vars = [
        'era5_tasmax_lag1', 'era5_tasmax_lag2',
        'era5_swvl1_lag5'
    ]

    for var_name in expected_vars:
        assert var_name in ds_lagged, f"Expected variable {var_name} not found in dataset."

    # 2. Check that original variables are preserved
    assert 'tasmax' in ds_lagged
    assert 'swvl1' in ds_lagged

    # 3. Verify the shifting logic for one specific lag (e.g., swvl1, lag 5)
    var = 'swvl1'
    lag = 5
    lag_var_name = f'{new_prefix}{var}_lag{lag}'

    original_flat = ds[var].values.flatten()
    lagged_flat = ds_lagged[lag_var_name].values.flatten()

    # Check the first 'lag' elements are NaN
    assert np.all(np.isnan(lagged_flat[:lag]))
    
    # Check the shift: lagged[lag:] should equal original[:-lag]
    assert np.allclose(lagged_flat[lag:], original_flat[:-lag], equal_nan=False)

# ==============================================================================
# Test features_labels.py
# ==============================================================================

def test_Compute_climatology_output_dims(dummy_xarray_dataset):
    """
    Tests if Compute_climatology returns the expected number of outputs 
    and if the core outputs (climatologies) have the correct 'dayofyear' dimension.
    """
    ds = dummy_xarray_dataset
    variable = 'tasmax'
    window = 5 # window for rolling mean
    # Use the dummy dataset's bounds for the reference period
    date1 = str(ds.time.min().dt.date.item())
    date2 = str(ds.time.max().dt.date.item())
    
    # Compute_climatology returns: climatology, clim_window, window_series, loess_climatology
    results = features_labels.Compute_climatology(ds, variable, window, date1, date2)
        
    # 1. Check number of returns
    assert len(results) == 4
    
    # Unpack for specific checks
    climatology, clim_window, window_series, loess_climatology = results
    
    # The expected dayofyear dimension is 365 since we used 2001-01-01 to 2002-12-31 (no leap days)
    expected_doy_dim = 365 
    
    # 2. Check Climatology doy dimension (presence and size)
    assert 'dayofyear' in climatology.dims
    assert climatology.dayofyear.size == expected_doy_dim
    
    # 3. Check Clim_Window (should also be averaged over time)
    assert 'dayofyear' in clim_window.dims
    assert clim_window.dayofyear.size == expected_doy_dim
    
    # 4. Check Window_Series (should retain the full time dimension)
    assert 'time' in window_series.dims
    assert window_series.time.size == ds.time.size
    
    # 5. Check Loess_Climatology (should have the same dayofyear dimension)
    assert 'dayofyear' in loess_climatology.dims
    assert loess_climatology.dayofyear.size == expected_doy_dim