# data_loading.py
import torch
import numpy as np
import yaml
import os
import random
import torch.nn as nn
import torch.optim as optim


from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions

def reset_seeds(g,seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)

def load_hypms_from_file(site_name, percentile='90p', base_path='/home/bsc/bsc167965/TFM/ML/HYPM_tunning_outputs',
                         file_name=None):
    """
    Loads hyperparameters for a given site and percentile from a text file. The hyperparameters to load are hardcoded.

    Args:
        site_name (str): The name of the site (e.g., 'cordoba').
        percentile (str): The percentile string, e.g., '95p' or '98p'.
        base_path (str): The directory containing the hyperparameter files.
        file_name (str): The name of the hyperparameter file. If None, it defaults to a standard naming convention.

    Returns:
        dict: A dictionary with the loaded hyperparameters or None if the file doesn't exist.
    """
    hypms = {}
    #file_path = os.path.join(base_path, file_name)

    # 1. Get the directory of the current script:
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. Go up two levels to reach the 'src' directory, then navigate down into the data files.
    # The path needs to be: /quantifydrivershw/src/quantifydrivers/data_files/HYPMS_optimization_results/

    # Path to 'quantifydrivers' directory
    quantifydrivers_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Construct the final path using os.path.join for reliability
    file_path = os.path.join(
        quantifydrivers_dir,
        "data_files",
        "HYPMS_optimization_results",
        f"g500_1lag_{site_name}_best_params_{percentile}_with_testing_phase.txt"
    )
    print(f"*** Checking hyperparameter file path: {file_path} ***") # NEW PRINT

    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None
    print(f"*** Hyperparameter file found for {site_name}. Loading content... ***") # NEW PRINT

    # This mapping handles differences between keys in the file and keys in script code
    key_mapping = {
        'extreme_weights_ctt': 'extreme_weights_ctt',
        'nonextreme_weights_ctt': 'nonextreme_weights_ctt'
    }

    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Use the mapped key if it exists, otherwise use the original key
                code_key = key_mapping.get(key, key)

                # Try to convert value to a number, skipping lines where this fails (like headers)
                try:
                    numeric_value = float(value)
                    if code_key == 'batch_size':
                        hypms[code_key] = int(numeric_value)
                    # Only add keys that are expected in the script
                    elif code_key in ['lr', 'w_decay', 'minority_weight_multiplier']:
                        hypms[code_key] = numeric_value
                except ValueError:
                    continue
    print(f"*** Hyperparameters successfully parsed. ***") # NEW PRINT
    return hypms



def training(config_path,datasets, seed, device, generator):
    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    g = generator
    SITE_HYPMS_fixed = {
        'belgrado': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                     'nonextreme_weights_ctt': 1},
        'hannover': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                     'nonextreme_weights_ctt': 1},
        'stockholm': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                      'nonextreme_weights_ctt': 1},
        'lyon': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1, 'nonextreme_weights_ctt': 1},
        'cordoba': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                    'nonextreme_weights_ctt': 1},
        'marrakech': {'lr': 1e-4, 'w_decay': 0.01, 'batch_size': 32, 'extreme_weights_ctt': 1,
                      'nonextreme_weights_ctt': 1}}

    # Create empty dictionary with the base HYPMS
    SITE_HYPMS = SITE_HYPMS_fixed.copy()

    # ==============================================================================================================
    # Choose which percentile's hyperparameters to load
    percentile_to_load = '90p'

    print(f"Loading hyperparameters for percentile: {percentile_to_load}")
    params = load_hypms_from_file(CONF["SITE"], percentile=percentile_to_load, file_name="file_with_hypms.txt")
    if params:
        SITE_HYPMS[CONF["SITE"]] = params
    print(f"Loaded hyperparameters for {CONF["SITE"]}: {SITE_HYPMS[CONF["SITE"]]}")

    print(f"Doing site: {CONF["SITE"]}")
    print(f"*** Setting up file paths for site: {CONF["SITE"]} ***")  # NEW PRINT

    # =================================================================================================================
    # Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
    # =================================================================================================================

    HYPMS = dict(
        epochs=75,
        lr=SITE_HYPMS[CONF["SITE"]]['lr'],
        w_decay=SITE_HYPMS[CONF["SITE"]]['w_decay'],
    )

    train_dataset = datasets["train_dataset"]
    test_dataset = datasets["test_dataset"]
    train_features_era5 = datasets["train_era5"]
    test_features_era5 = datasets["test_era5"]
    train_subset_combined = datasets["train_subset"]

    combined_train_loader = datasets["train_loader"]
    combined_val_loader = datasets["val_loader"]
    combined_test_loader = datasets["test_loader"]
    combined_test_dataset = datasets["combined_test"]

    number_lags = ['g500', 'g200', 'psl']

    # Name to save the trained CombinedModel
    name_save_CombinedModel = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"

    #  Weights class imbalance  ---------------------------------------------------------------------------------
    print("*** Calculating Class Weights ***")  # NEW PRINT

    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    print(f"*** Class counts (0: non-extreme, 1: extreme): {class_counts} ***")  # NEW PRINT
    total_counts = sum(class_counts)

    base_minority_weight = class_counts[0] / class_counts[1]

    minority_weight_multiplier = SITE_HYPMS[CONF["SITE"]]['minority_weight_multiplier']
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights
    print(f"*** Smoothed Class Weights (0, 1): {smoothed_weights.cpu().numpy()} ***")  # NEW PRINT

    # Cross-entropy loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=smoothed_weights)

    # Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
    print("*** Initializing Models (NN and CNN) ***")  # NEW PRINT

    reset_seeds(g,seed)
    # MLP for local-scale
    NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                           train_alone_NN=False, num_classes=2).to(device)
    reset_seeds(g,seed)
    # ConvNext for large-scale
    CNN_model_loaded = convnext_functions.ConvNext(
        num_channels=len(train_features_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device)
    reset_seeds(g,seed)

    # Combined model
    model = machine_learning.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                           output_dim=2).to(device)
    reset_seeds(g,seed)
    print("*** Combined Model initialized. Starting Training Phase... ***")  # NEW PRINT

    # =======================================================================================================================================
    # Train phase Combined model -------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================

    reset_seeds(g,seed)
    # Optimizer
    optimizer_combined = optim.AdamW(model.parameters(), lr=HYPMS['lr'], weight_decay=HYPMS['w_decay'])
    # Scheduler (if wanted)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_combined, T_max=30)

    # Start training
    print(" Training combined model ")
    losses_train_combined, losses_val_combined, num_e, best_val_loss = machine_learning.train_CombinedModel(model,
                                                                                                            combined_train_loader,
                                                                                                            combined_val_loader,
                                                                                                            criterion=criterion,
                                                                                                            optimizer=optimizer_combined,
                                                                                                            num_epochs=
                                                                                                            HYPMS[
                                                                                                                'epochs'],
                                                                                                            plot_loss=False,
                                                                                                            print_loss=False,
                                                                                                            early_stop=True,
                                                                                                            patience=5,
                                                                                                            print_early_stop=False,
                                                                                                            trial=None)

    # Save the trained CombinedModel -----------------------------
    save_path = f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{CONF["SITE"]}/trained_models/member_{seed}_{name_save_CombinedModel}_{CONF["SITE"]}_test_2.pth"

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # os.makedirs creates all intermediate folders
        print(f"Created output directory: {save_dir}")  # Optional: Confirmation print
    torch.save(model.state_dict(),
               f"/gpfs/scratch/bsc32/bsc214253/data/test_train_n_shap_dilation/{CONF["SITE"]}/trained_models/member_{seed}_{name_save_CombinedModel}_{CONF["SITE"]}_test_2.pth")
    print(f"*** Trained model state dictionary saved to: {save_path} ***")  # NEW PRINT

    return model, CNN_model_loaded, NN_model, losses_train_combined, losses_val_combined





