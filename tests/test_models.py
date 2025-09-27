import pytest
import torch
from quantifydrivers.machine_learning.models import CombinedModel
from quantifydrivers.machine_learning.convnext_functions import ConvNext
from quantifydrivers import machine_learning

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General Configuration -----------------------------------------------------
NN_INPUT_DIM = 4       # Corresponds to len(train_dataset.all_features)
CNN_INPUT_CHANNELS = 3 # Corresponds to len(train_features_era5.all_features)
OUTPUT_CLASSES = 2
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_input_data():
    """
    Creates dummy input tensors for the CombinedModel.
    We assume the model takes two inputs:
    1. A spatial/temporal input (e.g., climate data)
    2. A simple scalar/vector input (e.g., CO2)
    """
    # Shape: (batch_size, channels, height, width). What is passed to the CNN
    spatial_input = torch.randn(4, 5, 30, 30, dtype=torch.float32)
    
    # Simple input is a vector of 2 features (e.g., [CO2, swlv]). What is pased to the MLP
    # Shape: (batch_size, features)
    simple_input = torch.randn(4, 2, dtype=torch.float32)
    
    return spatial_input, simple_input

# --- Unit Tests ---

def test_convnext_initialization():
    """Checks if the ConvNext model can be initialized without errors, using only hardcoded parameters."""
    try:
        model = ConvNext(
           num_channels=CNN_INPUT_CHANNELS, # Using the hardcoded channel count (3)
           num_classes=OUTPUT_CLASSES,     # Using the hardcoded class count (2)
           patch_size=4,
           layer_dims=[4, 6, 6, 16],
           depths=[1, 2, 2, 1],
           drop_rate=0.05,
           train_alone=False,
        )
        assert isinstance(model, torch.nn.Module)
    except Exception as e:
        # If initialization fails for any reason (e.g., shape error in the constructor), 
        # the test fails gracefully.
        pytest.fail(f"ConvNext initialization failed: {e}")



def test_combinedmodel_output_shape(dummy_input_data):
    """
    Tests if the CombinedModel runs and produces the correct output shape.
    This version uses only dummy data and fixes the channel mismatch.
    """
    spatial_input, simple_input = dummy_input_data
    
    # Use the actual batch size and feature counts from the dummy data
    batch_size = spatial_input.shape[0]        
    spatial_channels = spatial_input.shape[1]  
    simple_features = simple_input.shape[1]    
    output_classes = 2                         

    # 1. Initialize the Component Models
    
    NN_model = machine_learning.ToCombineExtremeClassifier(
        input_dim=simple_features,            
        train_alone_NN=False, 
        num_classes=output_classes
    ).to(device)
    
    # CNN Model (receives spatial_input with 5 channels)
    CNN_model = ConvNext(
        num_channels=spatial_channels,        
        num_classes=output_classes,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device) 
    
    # 2. Initialize the Combined Model
    model = CombinedModel(
        nn_model=NN_model, 
        cnn_model=CNN_model, 
        nn_hidden_dim=8, 
        cnn_hidden_dim=16, 
        output_dim=output_classes
    ).to(device)
    
    # 3. Run a forward pass
    
    output = model(simple_input.to(device), spatial_input.to(device))
    
    # 4. Check the type and shape
    assert isinstance(output, torch.Tensor), "Output must be a PyTorch Tensor"
    
    # The expected shape for a classifier is (batch_size, output_classes)
    expected_shape = (batch_size, output_classes)
    
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"



def test_combinedmodel_trainable_parameters():
    """Checks if the CombinedModel has a significant number of trainable parameters."""

    # 1. Initialize Component Models using hardcoded dimensions
    NN_model = machine_learning.ToCombineExtremeClassifier(
        input_dim=NN_INPUT_DIM,
        train_alone_NN=False,
        num_classes=OUTPUT_CLASSES
    ).to(device)

    CNN_model = ConvNext(
        num_channels=CNN_INPUT_CHANNELS,
        num_classes=OUTPUT_CLASSES,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device)
    
    # 2. Initialize the Combined Model
    model = CombinedModel(
        NN_model, 
        CNN_model, 
        nn_hidden_dim=8, 
        cnn_hidden_dim=16, 
        output_dim=OUTPUT_CLASSES
    ).to(device)

    # 3. Assertion
    # Count only parameters that require gradients
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Assert that the model has a complex structure (more than 1000 parameters)
    assert num_params > 1000, f"Model has only {num_params} trainable parameters. Expected more than 1000."