# data_loading.py
import torch
import numpy as np
import yaml
import os
import random
import torch.nn as nn
import torch.optim as optim
import pickle
import shap

from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions


def reset_seeds(g,seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)



def compute_SHAP(configuration,datasets, generator, device):

    train_dataset = datasets["train_dataset"]
    train_features_era5 = datasets["train_era5"]
    train_subset_combined = datasets["train_subset"]
    combined_test_dataset = datasets["combined_test"]
    # =======================================================================================================================================
    # SHAP computation ----------------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================
    print("*** Starting SHAP Computation Phase ***")  # NEW PRINT

    # Name to save the trained CombinedModel

    # Prepare NN model and CNN model for SHAP -----------------------------------------------------------------------------------------------
    NN_model_loaded = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),train_alone_NN=False, num_classes=2).to(device)
    NN_model_loaded.eval()
    reset_seeds(generator,configuration["SEED"])
    CNN_model_loaded = convnext_functions.ConvNext(
        num_channels=len(train_features_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device)
    reset_seeds(generator,configuration["SEED"])

    # Create the Combined model for SHAP---------------------------------------------------------------------------------------------------
    model = machine_learning.CombinedModel(NN_model_loaded, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(generator,configuration["SEED"])
    number_lags = configuration["dataset_config"]["variables_era5"]
    model_name = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"

    model_dir = configuration["paths"]["model_dir"]
    weight_file = os.path.join(
        model_dir,
        configuration["SITE"],
        "trained_models",
        f"member_{configuration["SEED"]}_{model_name}_{configuration["SITE"]}_test_2.pth"
    )
    print("Loading model:", weight_file)

    print(f"*** Loading trained weights from: {weight_file} for SHAP ***")  # NEW PRINT
    model_state_dict = torch.load(weight_file,weights_only=True)
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

    reset_seeds(generator,configuration["SEED"])
    print("Initializing GradientExplainer...")
    explainer_grad = shap.GradientExplainer(model, background_data)
    print("Explainer initialized.")
    reset_seeds(generator,configuration["SEED"])
    print("Calculating SHAP values...")
    shap_values = explainer_grad.shap_values(explain_data)

    print(f"Finished computing SHAP values for SITE: {configuration["SITE"]}")  # CHANGED 'site' to 'SITE'

    # Select class to explaine, extreme (1) in our case ----------------------------------------------------------------
    class_index_to_explain = 1
    shap_values_nn_raw = shap_values[0][:, :, 1]  # NumPy array (N_explain, nn_features)
    shap_values_cnn_raw = shap_values[1][:, :, :, :, 1]  # NumPy array (N_explain, V, H, W)

    # Dictionary to save SHAP values -----------------------------------------------------------------------------------
    raw_shap_dict = {
        'nn': shap_values_nn_raw,
        'cnn': shap_values_cnn_raw,
    }

    #shap_dir = CONF["paths"]["shap_dir"]
    #out_file = os.path.join(shap_dir, SITE, f"shap_raw_{seed}.pkl")
    #os.makedirs(os.path.dirname(out_file), exist_ok=True)

    #with open(out_file, "wb") as f:
    #    pickle.dump(raw_shap_dict, f)

    #print(f"Finished training and SHAP value computing for site: {CONF["SITE"]}")  # CHANGED 'site' to 'SITE'

    return





