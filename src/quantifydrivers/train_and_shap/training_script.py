print("TRAINING PIPELINE IMPORTED")

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


def load_hypms_from_file(site_name, percentile='90p'):
    hypms = {}

    script_dir = os.path.dirname(os.path.realpath(__file__))
    quantifydrivers_dir = os.path.abspath(os.path.join(script_dir, '..'))

    file_path = os.path.join(
        quantifydrivers_dir,
        "data_files",
        "HYPMS_optimization_results",
        f"g500_1lag_{site_name}_best_params_{percentile}_with_testing_phase.txt"
    )
    print(f"*** Checking hyperparameter file path: {file_path} ***")  # NEW PRINT

    if not os.path.exists(file_path):
        print(f"Warning: Hyperparameter file not found for {site_name} at {file_path}")
        return None
    print(f"*** Hyperparameter file found for {site_name}. Loading content... ***")  # NEW PRINT

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
                code_key = key_mapping.get(key, key)
                try:
                    numeric_value = float(value)
                    if code_key == 'batch_size':
                        hypms[code_key] = int(numeric_value)
                    # Only add keys that are expected in the script
                    elif code_key in ['lr', 'w_decay', 'minority_weight_multiplier']:
                        hypms[code_key] = numeric_value
                except ValueError:
                    continue
    print(f"*** Hyperparameters successfully parsed. ***")  # NEW PRINT
    return hypms


def training(configuration,datasets, device, generator, timestamp):
    g = generator

    SITE_HYPMS_fixed = {'lr': configuration.hyperparameters.site_hypms.lr,
                        'w_decay': configuration.hyperparameters.site_hypms.w_decay,
                        'batch_size': configuration.hyperparameters.site_hypms.batch_size,
                        'minority_weight_multiplier': configuration.hyperparameters.site_hypms.minority_weight_multiplier}

    print(SITE_HYPMS_fixed['lr'], type(SITE_HYPMS_fixed['lr']))
    SITE_HYPMS = SITE_HYPMS_fixed.copy()

    params = None

    print(f"Loading hyperparameters for percentile: {configuration.percentile}")
    if not configuration.hyperparameters.default_hypms:
        params = load_hypms_from_file(configuration.site.name, percentile=configuration.percentile)
    if params:
        SITE_HYPMS = params

    print(f"Loaded hyperparameters for {configuration.site.name}: {SITE_HYPMS}")
    print(SITE_HYPMS_fixed['lr'], type(SITE_HYPMS_fixed['lr']))

    # =================================================================================================================
    # Dataset, Dataloaders and hyperparameter configuration ----------------------------------------------------------------------------------------------------------------------------
    # =================================================================================================================

    HYPMS = dict(
        epochs=75,
        lr=SITE_HYPMS['lr'],
        w_decay=SITE_HYPMS['w_decay'],
    )

    train_dataset = datasets["train_dataset"]
    train_features_era5 = datasets["train_era5"]

    combined_train_loader = datasets["train_loader"]
    combined_val_loader = datasets["val_loader"]


    #  Weights class imbalance  ---------------------------------------------------------------------------------
    print("*** Calculating Class Weights ***")  # NEW PRINT

    unique_classes, class_counts = np.unique(train_dataset.labels, return_counts=True)
    print(f"*** Class counts (0: non-extreme, 1: extreme): {class_counts} ***")  # NEW PRINT
    total_counts = sum(class_counts)

    base_minority_weight = class_counts[0] / class_counts[1]

    minority_weight_multiplier = SITE_HYPMS['minority_weight_multiplier']
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights
    print(f"*** Smoothed Class Weights (0, 1): {smoothed_weights.cpu().numpy()} ***")  # NEW PRINT

    # Cross-entropy loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=smoothed_weights)

    # Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
    print("*** Initializing Models (NN and CNN) ***")  # NEW PRINT

    reset_seeds(g,configuration.seed)

    # MLP for local-scale
    NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                           train_alone_NN=False, num_classes=2).to(device)

    reset_seeds(g,configuration.seed)

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

    reset_seeds(g,configuration.seed)

    # Combined model
    model = machine_learning.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,output_dim=2).to(device)
    reset_seeds(g,configuration.seed)
    print("*** Combined Model initialized. Starting Training Phase... ***")  # NEW PRINT

    # =======================================================================================================================================
    # Train phase Combined model -------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================
    reset_seeds(g,configuration.seed)

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
                                                                                                            HYPMS['epochs'],
                                                                                                            plot_loss=False,
                                                                                                            print_loss=False,
                                                                                                            early_stop=True,
                                                                                                            patience=5,
                                                                                                            print_early_stop=False,
                                                                                                            trial=None)

    # ---------------------------
    # SAVE MODEL
    # ---------------------------

    number_lags = configuration.dataset.variables_era5
    model_name = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
    model_dir = configuration.paths.model_dir
    save_path = os.path.join(
        model_dir,
        configuration.site.name,
        "trained_models",
        f"member_{configuration.seed}_{model_name}_{configuration.site.name}_test_2.pth"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    seed_results = {
        'losses_train': losses_train_combined,
        'losses_val': losses_val_combined
    }

    if configuration.dataset.use_spei:
        save_name = f"{configuration.site.name}_{configuration.percentile}_{configuration.seed}_{configuration.dataset.spei_spi}_losses.pkl"
    else:
        save_name = f"{configuration.site.name}_{configuration.percentile}_{configuration.seed}_losses.pkl"


    results_file = os.path.join(
        configuration.paths.results_dir,
        configuration.site.name,
        f"{configuration.site.name}_{configuration.percentile}_{configuration.seed}",
        save_name)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "wb") as f:
        pickle.dump(seed_results, f)





