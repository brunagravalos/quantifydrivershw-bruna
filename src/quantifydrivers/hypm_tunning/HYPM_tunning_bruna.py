
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
import pandas as pd
import torch
import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
import shap
from sklearn.metrics import balanced_accuracy_score
import pickle
import optuna
from sklearn.metrics import f1_score
import gc
import tqdm
import argparse

from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions



import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

from datetime import datetime


from quantifydrivers.train_and_shap.config_schema import validate_schema
from quantifydrivers.train_and_shap.dataloading_script import build_datasets_and_loaders

# SETS DETERMINISM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device set to: {device} ***")

try:
    torch.use_deterministic_algorithms(True)
    print("Using deterministic algorithms.")
except Exception as e:
    print(f"Could not enforce deterministic algorithms: {e}")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def verify_determinism():
    # Check PyTorch
    print(f"PyTorch rand(): {torch.rand(1).item()}")

    # Check NumPy
    print(f"NumPy rand(): {np.random.rand()}")

    # Check Python random
    print(f"Python random(): {random.random()}")


# Function to Reset the seed, for determinsim in the computations *********************************************************************************************************

def reset_seeds(seed=42):
    g = torch.Generator()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g.manual_seed(seed)


# Generate list of seeds for the ensamble loop ***************************************************************************************************************************

def generate_ensemble_seeds(fixed_seed=123):
    rng = np.random.default_rng(fixed_seed)
    seeds = rng.integers(low=0, high=2 ** 32 - 1, size=1).tolist()
    return seeds


def load_mock_paths(base_folder: str, site: str, percentile: str):
    """
    base_folder points to the Zarr root, e.g. /data/climate_data.zarr
    """
    return {
        "g500": os.path.join(base_folder, "era5", "g500"),
        "g200": os.path.join(base_folder, "era5", "g200"),
        "psl":  os.path.join(base_folder, "era5", "psl"),
        "co2":  os.path.join(base_folder, "aux", "co2"),
        "local": os.path.join(base_folder, "era5land", percentile, site),
    }

def objective(trial, n_trials, configuration, datasets):
    # 1. Hyperparameters for this trial
    # =================================================================================
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    w_decay = trial.suggest_float("w_decay", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    minority_weight_multiplier = trial.suggest_float("minority_weight_multiplier", 1., 15.0)

    reset_seeds(configuration.seed)

    # =================================================================================================
    # Dataloaders setup
    # =================================================================================
    combined_train_loader = datasets["train_loader"]
    combined_val_loader = datasets["val_loader"]

    # 3. Setup Model, Optimizer and Loss with suggested hyperparameters
    # =================================================================================
    # Class weights
    unique_classes, class_counts = np.unique(datasets["train_dataset"].labels, return_counts=True)
    base_minority_weight = class_counts[0] / class_counts[1]
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=smoothed_weights, reduction='mean').to(device)

    # Initialize models
    reset_seeds(configuration.seed)
    NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(datasets["train_dataset"].all_features),
                                                           train_alone_NN=False, num_classes=2).to(device)
    CNN_model = convnext_functions.ConvNext(
        num_channels=len(datasets["train_era5"].all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False
    ).to(device)

    model = machine_learning.CombinedModel(NN_model, CNN_model, nn_hidden_dim=8, cnn_hidden_dim=16, output_dim=2).to(
        device)

    # Optimizer
    optimizer_combined = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)
    # Learning rate scheduler if desired
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_combined, T_max=30)  # 30 is num_epochs

    # Mixing 32, 16 bit float for speed-up with scaler
    scaler = torch.cuda.amp.GradScaler()
    # =================================================================================
    # 4. Train and Evaluate the model
    # =================================================================================
    print("Training combined model...")
    reset_seeds(configuration.seed)

    losses_train_combined, losses_val_combined, num_e, best_val_loss = machine_learning.train_CombinedModel(model,
                                                                                                            combined_train_loader,
                                                                                                            combined_val_loader,
                                                                                                            criterion=criterion,
                                                                                                            optimizer=optimizer_combined,
                                                                                                            num_epochs=50,
                                                                                                            plot_loss=False,
                                                                                                            print_loss=False,
                                                                                                            early_stop=True,
                                                                                                            patience=5,
                                                                                                            print_early_stop=True,
                                                                                                            trial=trial)

    # Evaluate on the validation set to get the score for this trial
    y_true_val, y_pred_val, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
        CombinedModel=model,
        cnn=CNN_model,
        nn=NN_model,
        test_loader=combined_val_loader,
        print_accuracies=True,
        train_alone=False
    )

    final_val_loss = best_val_loss  # use the best val loos found before early stoping

    # =================================================================================
    # 5. Define and Return the Score to be Optimized
    # =================================================================================

    balanced_accuracy = balanced_accuracy_score(y_true_val, y_pred_val)
    # Use balanced accuracy minus validation loss as the score
    score = balanced_accuracy - final_val_loss

    return score


def main():

    sites = ['cordoba', 'lyon', 'hannover', 'stockholm', 'belgrado', 'marrakech']

    best_params_per_site = {}

    rel_config_path = "../train_and_shap/conf"

    # Start hyperparameter tuning for each site -----------------------------------------------------------------------------------

    for site in sites:
        # Initialize & validate config
        with initialize(version_base=None, config_path=rel_config_path):
            cfg = compose(config_name="config", overrides=[f"site={site}"])
        try:
            configuration = validate_schema(cfg)
            print("Config Validation Passed!")

        except Exception as e:
            print("Config Validation Failed or validate_schema not imported.")

        print(f"Doing site: {site}")
        g = torch.Generator()
        g.manual_seed(configuration.seed)

        # --- Build datasets
        datasets = build_datasets_and_loaders(configuration=configuration, generator=g)

        print(f"\n--- Starting Hyperparameter Tuning for site: {configuration.site} ---")

        # The lambda function is used to pass extra arguments (site, seed) to the objective function
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))  # Maximize the score defined!
        study.optimize(lambda trial: objective(trial, n_trials=20, configuration=configuration, datasets= datasets))  # Run n trials

        # Store the best parameters found for the site
        best_params = study.best_trial.params
        best_params_per_site[site] = best_params

        print(f"--- Tuning Finished for {site} ---")

        best_trial = study.best_trial

        print(f"Best score: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        results_content = f"Best hyperparameters for site: {site}\n"
        results_content += f"Best trial score: {best_trial.value:.4f}\n\n"
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                line = f"{key}: {value:.6f}\n"
            else:
                line = f"{key}: {value}\n"
            print(f"    {line.strip()}")
            results_content += line

        # Save the best parameters to a text file
        output_dir = configuration.paths.hyperparameters_dir
        file_path = os.path.join(output_dir, f"hyperparameters_{site}_{configuration.percentile}.txt")

        with open(file_path, 'w') as f:
           f.write(results_content)



if __name__ == "__main__":
    main()
