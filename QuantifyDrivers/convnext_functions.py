
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 

from torchvision.ops import stochastic_depth

class PaddedStem(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        H, W = x.shape[-2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant',value=0)  # Pad (left, right, top, bottom)
        x = self.conv(x)
        x = self.norm(x)
        return x


class PaddedDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        H, W = x.shape[-2:]
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = self.norm(x)
        x = self.conv(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class ConvNextBlock(nn.Module):

    

    def __init__(self, filter_dim, layer_scale=1e-6): #, dilation=1, padding_mode='zeros'): #original does not have dilation and padding_mode as arguments 
        super().__init__()

        kernel_size = 7 

        #current_padding = dilation * (kernel_size - 1) // 2 # formula for "same" padding, to mantain spatial dimensions

        self.block = nn.Sequential(*[
            nn.Conv2d(filter_dim,
                      filter_dim,
                      kernel_size=7,
                      padding=3,       #current_padding, # 3 for original. Padding 3 chosen to mantain spatial dimensions after convolution
                      groups=filter_dim,
                      #dilation = dilation,
                      ),
                
            Permute([0, 2, 3, 1]),
            LayerNorm(filter_dim, eps=1e-6),
            nn.Linear(filter_dim, filter_dim * 4),
            nn.GELU(),
            nn.Linear(filter_dim * 4, filter_dim),
            Permute([0, 3, 1, 2])
        ])
        self.gamma = nn.Parameter(torch.ones(filter_dim, 1, 1) * layer_scale)

    def forward(self, x):
        return self.block(x) * self.gamma


class ConvNextLayer(nn.Module):

    def __init__(self, filter_dim, depth, drop_rates): #, dilation=1, block_padding_mode='zeros'): #original does not have dilation or block_padding
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(ConvNextBlock(filter_dim=filter_dim  # original only has filter_dim as argument
                                            #layer_scale=1e-6, 
                                            #dilation=dilation, 
                                            ))

        self.drop_rates = drop_rates

    def forward(self, x):
        for idx, block in enumerate(self.blocks):

            if self.drop_rates[idx] == 0.0:
                x = x + block(x)
            else:
                x = x + stochastic_depth(block(x),
                                         self.drop_rates[idx],
                                         mode="batch",
                                         training=self.training)
    
            #x = x + stochastic_depth(block(x),
            #                         self.drop_rates[idx],
            #                         mode="batch",
            #                         training=self.training)
        return x


class ConvNext(nn.Module):

    def __init__(self,
                 num_channels=3,
                 num_classes=10,
                 patch_size=4,
                 layer_dims=[96, 192, 384, 768],
                 depths=[3, 3, 9, 3],
                 drop_rate=0.,
                 train_alone=True):
        super().__init__()

        self.train_alone = train_alone

        # init downsample layers with stem
        #self.downsample_layers = nn.ModuleList(
        #    [nn.Sequential(
        #        nn.Conv2d(num_channels, layer_dims[0], kernel_size=patch_size, stride=patch_size),
        #        LayerNorm(layer_dims[0],
        #                      eps=1e-6,
        #                      data_format="channels_first")
        #    )])

        self.downsample_layers = nn.ModuleList([
            PaddedStem(num_channels, layer_dims[0], patch_size)
        ])
        
        for idx in range(len(layer_dims) - 1):
           #self.downsample_layers.append(
           #    nn.Sequential(
           #        LayerNorm(layer_dims[idx],
           #                  eps=1e-6,
           #                  data_format="channels_first"),
           #        nn.Conv2d(layer_dims[idx],
           #                  layer_dims[idx + 1],
           #                  kernel_size=2,
           #                  stride=2
           #                  ),
           #    ))

            self.downsample_layers.append(
                PaddedDownsample(layer_dims[idx], layer_dims[idx + 1])
            )

        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))

        if self.train_alone:
            self.cls = nn.Sequential(
                LayerNorm(layer_dims[-1], eps=1e-6),
                nn.Linear(layer_dims[-1], num_classes)    
            )

        if not self.train_alone:
            self.spatial_aggregator = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Reduces spatial dims to 1x1
                nn.Flatten(),
                nn.Linear(layer_dims[-1], 16)  # Project to desired hidden dim
            )

    def forward(self, x):
        all_layers = list(zip(self.downsample_layers, self.stage_layers))
        for downsample_layer, stage_layer in all_layers:
            x = downsample_layer(x)
            x = stage_layer(x)

        #if self.train_alone:
        #    return self.cls(x.mean(dim=(-2, -1)))
        #else:
        #    return self.spatial_aggregator(x)  # Use learnable aggregation
     
        if self.train_alone:
            return self.cls(x.mean(dim=(-2, -1)))
        else:
            return x.mean(dim=(-2, -1))
    

# Sequential ConvNeXt class .....................................................................................

class ConvNextLagProcessor(nn.Module): # Renamed for clarity

    def __init__(self,            
                 num_variables,   
                 num_lags,             
                 num_classes=2,     
                 
                 patch_size=4,        
                 layer_dims=[96, 192, 384, 768], 
                 depths=[3, 3, 9, 3],  
                 drop_rate=0.,         
                
                 train_alone=True,
                 combined_hidden_dim=64
                 ):
        super().__init__()

        self.num_variables = num_variables
        self.num_lags = num_lags
        self.layer_dims = layer_dims 
        self.train_alone = train_alone

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(num_variables, layer_dims[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(layer_dims[0], eps=1e-6, data_format="channels_first")
            ))
        for idx in range(len(layer_dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(layer_dims[idx], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(layer_dims[idx], layer_dims[idx + 1], kernel_size=2, stride=2),
                ))

        drop_rates = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))
       
        pooled_dim_per_lag = layer_dims[-1]
        combined_input_dim = self.num_lags * pooled_dim_per_lag

        # FC layer to process the concatenated features from all lags
        self.fc_combine_lags = nn.Sequential(
            nn.Linear(combined_input_dim, combined_hidden_dim),
            # Using ReLU but could use GELU like ConvNextBlock
            nn.ReLU()
            # Final Layernorm could be added
            # nn.LayerNorm(combined_hidden_dim, eps=1e-6)
        )

        if self.train_alone:
            self.final_classification = nn.Sequential(
                 nn.LayerNorm(combined_hidden_dim, eps=1e-6),
                 nn.Linear(combined_hidden_dim, num_classes)
            )

    def forward(self, x):
        # x : (B, V, L, H, W)
        batch_size = x.size(0)
        lag_outputs_pooled = [] 

        # Iterate through each time lag (dimension index 2)
        for t in range(self.num_lags):
            # Extract the data for the current lag t
            lag_data = x[:, :, t, :, :]

            x_lag = lag_data
            all_layers = list(zip(self.downsample_layers, self.stage_layers))
            for downsample_layer, stage_layer in all_layers:
                x_lag = downsample_layer(x_lag)
                x_lag = stage_layer(x_lag)
          
            pooled_out = x_lag.mean(dim=(-2, -1))
            lag_outputs_pooled.append(pooled_out)

       
        combined_output = torch.cat(lag_outputs_pooled, dim=1)

        hidden = self.fc_combine_lags(combined_output)

        # Return classification logits or hidden features based on the flag
        if self.train_alone:
            return self.final_classification(hidden)
        else:
            return hidden 


