import torch
import scipy 
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import torch.nn as nn                   # provides classes and functions to create and train neural networks
import torch.nn.functional as F         # provides functions for activation functions, loss functions, and other operations


class Mask_pretask_CNN(nn.Module):

    """ 
    Input data is modified and CNN needs to reconstruct it 
    """
    
    def __init__(self, cnn_model, num_variables, num_lags):
        super().__init__()
        self.cnn = cnn_model 
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, 1)) # create learnable parameter. Will replaced masked regions of the input 
        self.reconstruction_head = nn.Sequential( # layers to reconstruct the input 
            nn.Conv2d(128, 64, kernel_size=1), 
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)   
        )

        self.mask_pool = nn.AdaptiveAvgPool2d((7, 15)) # pool to match shape after convolution blocks 

        self.num_vars = num_variables
        self.num_lags = num_lags

    def forward(self, x, mask_ratio=0.15):
        """ 
        x : input (B,features,lat,lon)
        mask_ratio: % of the input that will be masked to reconstruct """
        # x dimensions --> (batch, 3, lat, lon) [3 anomaly variables]

        b, c, lags, h, w = x.shape
        
        masked_x = x.clone()
        mask = torch.rand(x.shape[0], 1, 1,x.shape[3], x.shape[4]) < mask_ratio #one channel tensor with values between 0 and 1 and create mask comparing with mask_ratio 
        masked_x[mask.expand(-1, self.num_vars, self.num_lags, -1,-1)] = self.mask_token  # expand to the three channels. All true values are replaced with learnable tensor. 

        # This is masking all variables in the same cells. Could do variable specific masking as well. 
        
        recon_per_lag = []
        for lag in range(self.num_lags):
            features = self.cnn(masked_x[:, :, lag, :, :])  # [batch, 128, 7, 15]
            recon = self.reconstruction_head(features)  # [B, 3, 7, 15]
            recon_per_lag.append(recon)
        
        recon = torch.stack(recon_per_lag, dim=2)  # [batch, num_vars, lag, 7, 15]

        x_downsampled = F.max_pool2d(
            x.view(b*c*lags, 1, h, w),  
            kernel_size=4, stride=4
        )
        
        x_downsampled = x_downsampled.view(b, c, lags, x_downsampled.shape[2], x_downsampled.shape[3]) 
        
        mask_downsampled = F.max_pool2d(
        mask.float().view(b, 1, h, w),  # Convert to 4D [B,1,H,W]
        kernel_size=4, stride=4
        ).bool().view(b, 1, 1, x_downsampled.shape[3], x_downsampled.shape[4]).expand(-1, c, lags, -1, -1)
        
        return recon, x_downsampled, mask_downsampled #return reconstruction and mask



# Loss functions ----------------------------------------------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: balancing factor for class imbalance (scalar)
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits (before sigmoid), shape [N]
        targets: ground truth labels, shape [N], values in {0,1}
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Spatial Atention ---------------------------------------------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # project to 1 attention map

    def forward(self, x):
        # x dimentions: [batch, features, lat, lon]
        attn = self.conv(x)                # [B, 1, H, W]
        attn = torch.sigmoid(attn)         # [0,1] range attention weights
        return x * attn                    # element-wise multiplication



# Interaction class

class InteractionLayer(nn.Module):
    def __init__(self, nn_dim: int, cnn_dim: int, hidden_dim: int ):
        """
        Args:
            nn_dim: Dimension of tabular features (from ToCombineExtremeClassifier)
            cnn_dim: Dimension of CNN features (from CNN_Era5ExtemeClassifer)
            hidden_dim:  attention network size
        """
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU() 
        )
        
        common_dim = 128 
        
        self.nn_proj = nn.Linear(nn_dim, common_dim)
        self.cnn_proj = nn.Linear(cnn_dim, common_dim) 

    def forward(self, nn_feat: torch.Tensor, cnn_feat: torch.Tensor) -> tuple:
        """
        Args:
            nn_feat: Tabular features [batch, nn_dim]
            cnn_feat: Spatial features [batch, cnn_dim]
            
        Returns:
            tuple: (interacted_nn, interacted_cnn)
        """
        nn_proj = self.nn_proj(nn_feat)
        cnn_proj = self.cnn_proj(cnn_feat)
        
        combined = torch.cat([nn_proj, cnn_proj], dim=1)
        attention = self.attention_net(combined)  # learns how much to focus on each source
        
        interacted_nn = attention * nn_proj
        interacted_cnn = (1 - attention) * cnn_proj
        
        return interacted_nn, interacted_cnn





class ERA5Dataset_extremes_pretrain(Dataset):
    def __init__(self, file_g500,file_g200,file_psl, start_date, end_date, months, lags_era5,transform=None):
        """
        Args:
            file_g500 (str): Path to the NetCDF file for g500 EOFs.
            file_g200 (str): Path to the NetCDF file for g200 EOFs.
            file_psl (str): Path to the NetCDF file for psl EOFs.
            start_date (str): Start date for filtering the dataset.
            end_date (str): End date for filtering the dataset.
            months (list of int): List of months to filter.
            scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler for standardization.
            transform (callable, optional): Optional transform to be applied.
        """

        # Select time period and months 
        self.ds_g500 = xr.open_dataset(file_g500).sel(time=slice(start_date, end_date))
        self.ds_g500 = self.ds_g500.sel(time=self.ds_g500.time.dt.month.isin(months), drop=True)
        
        self.ds_g200 = xr.open_dataset(file_g200).sel(time=slice(start_date, end_date))
        self.ds_g200 = self.ds_g200.sel(time=self.ds_g200.time.dt.month.isin(months), drop=True)
        self.ds_g200 = self.ds_g200.squeeze('plev', drop=True) #remove plev dimension
        
        self.ds_psl = xr.open_dataset(file_psl).sel(time=slice(start_date, end_date))
        self.ds_psl = self.ds_psl.sel(time=self.ds_psl.time.dt.month.isin(months), drop=True)

        self.lagged_vars = {
            'g500': ['lagged_era5g500_anomalies_lag1'],
            'g200': ['lagged_era5g200_anomalies_lag1'],
            'psl':  ['lagged_era5psl_anomalies_lag1'],
        }
       
        self.lagged_vars_g500 = self.lagged_vars['g500']
        self.lagged_vars_g200 = self.lagged_vars['g200']
        self.lagged_vars_psl = self.lagged_vars['psl']

        vars_names = self.lagged_vars_g500 + self.lagged_vars_g200 + self.lagged_vars_psl 
            
        self.all_features = vars_names
        
        
        self.features_g500 = self.ds_g500[self.lagged_vars_g500].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')
        self.features_g200 = self.ds_g200[self.lagged_vars_g200].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')
        self.features_psl = self.ds_psl[self.lagged_vars_psl].to_array(dim='lag').transpose('time', 'lag', 'lat', 'lon')
        

        self.features = np.concatenate(
            [self.features_g500.values, self.features_g200.values, self.features_psl.values],axis=1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.features[idx], dtype=torch.float32)  
        
        return sample




class Mask_pretask_CNN_1lag(nn.Module):

    """ 
    Input data is modified and CNN needs to reconstruct it 
    """
    
    def __init__(self, cnn_model):
        super().__init__()
        self.cnn = cnn_model 
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, 1)) # create learnable parameter. Will replaced masked regions of the input 
        self.reconstruction_head = nn.Sequential( # layers to reconstruct the input 
            nn.Conv2d(128, 64, kernel_size=1), 
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)   
        )

        self.mask_pool = nn.AdaptiveAvgPool2d((7, 15)) # pool to match shape after convolution blocks 

    def forward(self, x, mask_ratio=0.15):
        """ 
        x : input (B,features,lat,lon)
        mask_ratio: % of the input that will be masked to reconstruct """
        # x dimensions --> (batch, 3, lat, lon) [3 anomaly variables]
        masked_x = x.clone()
        mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) < mask_ratio #one channel tensor with values between 0 and 1 and create mask comparing with mask_ratio 
        masked_x[mask.expand(-1, 3, -1, -1)] = self.mask_token  # expand to the three channels. All true values are replaced with learnable tensor. 

        # This is masking all variables in the same cells. Could do variable specific masking as well. 
        
        features = self.cnn(masked_x)  # Extract features
        recon = self.reconstruction_head(features)  # Reconstruct original

        mask_downsampled = F.max_pool2d(mask.float(), kernel_size=4, stride=4)  # [B, 1, 7, 15]
        mask_downsampled = mask_downsampled.bool()

        x_downsampled = F.max_pool2d(x, kernel_size=4, stride=4)
        
        mask_downsampled = mask_downsampled.expand(-1, 3, -1, -1) # back to true and fallse and expand to the three input channels 
        return recon, x_downsampled, mask_downsampled #return reconstruction and mask

