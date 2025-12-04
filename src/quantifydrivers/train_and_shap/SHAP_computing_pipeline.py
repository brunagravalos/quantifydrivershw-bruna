# data_loading.py
import torch
import numpy as np
import yaml
import os
import random
import torch.nn as nn
import torch.optim as optim
import pickle

from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions

def reset_seeds(g,seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)



def evaluation(config_path,datasets, seed, generator, losses_train_combined, losses_val_combined, model,CNN_model_loaded, NN_model):
    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    g = generator

    combined_test_dataset = datasets["data"]
    # Prepare NN model and CNN model for SHAP -----------------------------------------------------------------------------------------------
    NN_model_loaded = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                                  train_alone_NN=False, num_classes=2).to(device)
    NN_model_loaded.eval()
    reset_seeds(seed)
    CNN_model_loaded = convnext_functions.ConvNext(
        num_channels=len(train_features_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device)
    reset_seeds(seed)

    # Create the Combined model for SHAP---------------------------------------------------------------------------------------------------
    model = machine_learning.CombinedModel(NN_model_loaded, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                           output_dim=2).to(device)
    reset_seeds(seed)
    # Load the trained CombinedModel weights ----------------------------------------------------------------------------------------------
    save_path = f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{seed}_{name_save_CombinedModel}_{SITE}_test_2.pth"

    print(f"*** Loading trained weights from: {save_path} for SHAP ***")  # NEW PRINT
    model_state_dict = torch.load(
        f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{SITE}/trained_models/member_{seed}_{name_save_CombinedModel}_{SITE}_test_2.pth",
        weights_only=True)
    model.load_state_dict(model_state_dict)
    model.eval()

    # ---------------------------------------------------------------------------------------------------------------------

    # Create baseline for SHAP computation --------------------------------------------------------------------------------
    print(f"*** Creating SHAP background dataset (200 samples) ***")  # NEW PRINT
    background_indices = np.random.choice(len(train_subset_combined), 200, replace=False)
    background_nn = []
    background_cnn = []

    for idx in background_indices:
        nn_input, cnn_input, labels = train_subset_combined[idx]  # Adjust based on your dataset structure
        background_nn.append(nn_input)
        background_cnn.append(cnn_input)

    device = next(model.parameters()).device

    background_nn = torch.stack(background_nn, dim=0).to(device)
    background_cnn = torch.stack(background_cnn, dim=0).to(device)

    # Create explainer dataset for SHAP computation -----------------------------------------------------------------------
    print(f"*** Creating SHAP explanation dataset ({len(combined_test_dataset)} samples) ***")  # NEW PRINT
    explainer_indices = np.arange(0, len(combined_test_dataset), 1)
    explain_nn = []
    explain_cnn = []

    for idx in explainer_indices:
        nn_input, cnn_input, labels = combined_test_dataset[idx]
        explain_nn.append(nn_input)
        explain_cnn.append(cnn_input)

    explain_nn = torch.stack(explain_nn, dim=0).to(device)
    explain_cnn = torch.stack(explain_cnn, dim=0).to(device)

    # Merge local-scale and large-scale data for SHAP computation ------------------------------------------------------
    background_data = [background_nn, background_cnn]
    explain_data = [explain_nn, explain_cnn]

    reset_seeds(seed)
    print("Initializing GradientExplainer...")
    explainer_grad = shap.GradientExplainer(model, background_data)
    print("Explainer initialized.")
    reset_seeds(seed)
    print("Calculating SHAP values...")
    shap_values = explainer_grad.shap_values(explain_data)

    print(f"Finished computing SHAP values for SITE: {CONF["SITE"]}")  # CHANGED 'site' to 'SITE'

    # Select class to explaine, extreme (1) in our case ----------------------------------------------------------------
    class_index_to_explain = 1
    shap_values_nn_raw = shap_values[0][:, :, 1]  # NumPy array (N_explain, nn_features)
    shap_values_cnn_raw = shap_values[1][:, :, :, :, 1]  # NumPy array (N_explain, V, H, W)

    # Dictionary to save SHAP values -----------------------------------------------------------------------------------
    raw_shap_dict = {
        'nn': shap_values_nn_raw,
        'cnn': shap_values_cnn_raw,
    }

    # with open(f'/your/path/to/save/SHAP/results', 'wb') as f:
    #   pickle.dump(raw_shap_dict, f)

    print(f"Finished training and SHAP value computing for site: {CONF["SITE"]}")  # CHANGED 'site' to 'SITE'

    return





