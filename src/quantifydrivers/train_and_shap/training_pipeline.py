# data_loading.py
import torch
import numpy as np
import yaml
import os

from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training(config_path,datasets, seed,device):
    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

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

    minority_weight_multiplier = SITE_HYPMS[SITE]['minority_weight_multiplier']
    final_minority_weight = base_minority_weight * minority_weight_multiplier
    class_weights = torch.tensor([1.0, final_minority_weight], dtype=torch.float).to(device)
    smoothed_weights = torch.sqrt(class_weights).to(device)  # smoothing the weights
    print(f"*** Smoothed Class Weights (0, 1): {smoothed_weights.cpu().numpy()} ***")  # NEW PRINT

    # Cross-entropy loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=smoothed_weights)

    # Final training of the combined model -------------------------------------------------------------------------------------------------------------------------------------------------
    print("*** Initializing Models (NN and CNN) ***")  # NEW PRINT

    reset_seeds(seed)
    # MLP for local-scale
    NN_model = machine_learning.ToCombineExtremeClassifier(input_dim=len(train_dataset.all_features),
                                                           train_alone_NN=False, num_classes=2).to(device)
    reset_seeds(seed)
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
    reset_seeds(seed)

    # Combined model
    model = machine_learning.CombinedModel(NN_model, CNN_model_loaded, nn_hidden_dim=8, cnn_hidden_dim=16,
                                           output_dim=2).to(device)
    reset_seeds(seed)
    print("*** Combined Model initialized. Starting Training Phase... ***")  # NEW PRINT

    # =======================================================================================================================================
    # Train phase Combined model -------------------------------------------------------------------------------------------------------------
    # =======================================================================================================================================

    reset_seeds(seed)
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





