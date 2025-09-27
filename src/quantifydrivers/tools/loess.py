
# ==========================================================
# IMPORT NEEDED PACKAGES
# ==========================================================

import numpy as np
import time
import math

# ==========================================================

# ==========================================================
# This script implements LOESS (Locally Estimated Scatterplot Smoothing)
# for smoothing time series and gridded climate data.
#
# Main functionalities:
#   - loess_ts: Apply LOESS smoothing to a 1D time series.
#   - loess_3d: Apply LOESS smoothing along the time dimension of 3D data (time, lat, lon).
#   - tricubic: Weighting kernel used in LOESS regression.
#   - Loess class: Core implementation of LOESS regression, including:
#       * Normalization and denormalization of inputs
#       * Selection of nearest neighbors for local fitting
#       * Weighted regression for linear or polynomial fitting
#
# Dependencies: numpy, math, time
# ==========================================================


def loess_ts(ts,na_rm, window, degree):
    
    """ Function that performs a loess smoothing on a time series.
        The time window is set to 30 points per month.
        input: ts: time series (xarray object)
                na_rm: if True, removes nan values from the time series during fitting
                window: number of points to use for the loess smoothing 
        output: ts_loess: smoothed time series (xarray object)
    """
    ts = np.array(ts)
    if np.isnan(ts).all():
        return np.full(ts.shape, np.nan)
    elif not np.isnan(ts).any() or na_rm:
        # drop nan values
        x = range(len(ts))
        loess = Loess(range(len(ts[~np.isnan(ts)])), 
                            ts[~np.isnan(ts)], 
                            degree=degree)
        ts_loess = []
        for xx in x:
            ts_loess.append(loess.estimate(xx, window=window, degree=degree))
    else:
        exit("Error: missing values in the time series and na_rm=False")
    
            
    return np.array(ts_loess)

def loess_3d(ts, na_rm, window, degree):
    """
    Function that performs a LOESS smoothing along the first dimension (time) 
    for a 3D array (time, latitude, longitude).
    
    Parameters:
        ts (np.array): 3D input array with dimensions (time, latitude, longitude)
        na_rm (bool): If True, removes NaN values during fitting
        window (int): Number of points to use for the LOESS smoothing
        degree (int): Degree of the polynomial for LOESS fitting
    
    Returns:
        np.array: Smoothed 3D array with the same shape as the input
    """
    ts = np.array(ts)  # Ensure input is a NumPy array
    smoothed_ts = np.full_like(ts, np.nan)  # Initialize output array with NaNs
    
    time_len, lat_len, lon_len = ts.shape  # Extract dimensions
    
    for lat in range(lat_len):
        for lon in range(lon_len):
            ts_series = ts[:, lat, lon]
            
            if np.isnan(ts_series).all():
                continue  # Skip if all values are NaN
            elif not np.isnan(ts_series).any() or na_rm:
                x = np.arange(len(ts_series))
                valid_mask = ~np.isnan(ts_series)
                
                loess = Loess(x[valid_mask], ts_series[valid_mask], degree=degree)
                
                smoothed_ts_series = [loess.estimate(xx, window=window, degree=degree) for xx in x]
                smoothed_ts[:, lat, lon] = smoothed_ts_series
            else:
                raise ValueError("Error: Missing values in the time series and na_rm=False")
    
    return smoothed_ts


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)
