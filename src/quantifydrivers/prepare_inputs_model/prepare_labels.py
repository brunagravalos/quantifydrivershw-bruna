
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
from functions import data_preprocess as dp
import functions import loess_functions

# =========================================================

