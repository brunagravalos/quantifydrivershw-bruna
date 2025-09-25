# ======================================================================================================
# IMPORT NEEDED PACKAGES
# ======================================================================================================

import torch
import scipy 
import xarray as xr
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import functions_improve_CombinedModel
import optuna 
import random
from tqdm import tqdm
import torch.nn as nn                   
import torch.nn.functional as F  

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
    

# -------------------------------------------------------------------------------------------------

# Extract features and labels from the loaded dataset for the model ---------------------------------------------------------------------------------------------------------------------
def get_features_labels(ds,all_features):
    """ Given a dataset, extracts features and labels for a extreme, non-extreme classification
    """
    features = ds[all_features].to_array(dim='feature').transpose('time', 'feature').values   
    labels = ds['tasmax_extreme_classification'].values                                         
    valid_indices = ~np.isnan(features).any(axis=1)                                           
    return features[valid_indices], labels[valid_indices]
# -------------------------------------------------------------------------------------------------



# =============================================================================================================================
# END
# =============================================================================================================================

