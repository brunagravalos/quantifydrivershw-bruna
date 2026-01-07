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


def evaluation(configuration, datasets, generator, device, timestamp):

    print("*** Starting Evaluation Phase on Test Data ***")

    SITE = configuration.site
    seed = configuration.seed
    percentile = configuration.percentile
    g = generator

    # -----------------------------
    # 1. Rebuild model architecture
    # -----------------------------
    train_dataset = datasets["train_dataset"]
    train_era5 = datasets["train_era5"]

    # MLP local-scale
    reset_seeds(g, seed)
    NN_model = machine_learning.ToCombineExtremeClassifier(
        input_dim=len(train_dataset.all_features),
        train_alone_NN=False,
        num_classes=2
    ).to(device)

    # CNN large-scale
    reset_seeds(g, seed)
    CNN_model_loaded = convnext_functions.ConvNext(
        num_channels=len(train_era5.all_features),
        num_classes=2,
        patch_size=4,
        layer_dims=[4, 6, 6, 16],
        depths=[1, 2, 2, 1],
        drop_rate=0.05,
        train_alone=False,
    ).to(device)

    # Combined model
    reset_seeds(g, seed)
    model = machine_learning.CombinedModel(
        NN_model, CNN_model_loaded,
        nn_hidden_dim=8,
        cnn_hidden_dim=16,
        output_dim=2
    ).to(device)

    # -----------------------------
    # 2. Load saved model weights
    # -----------------------------
    number_lags = configuration.dataset.variables_era5
    model_name = f"CO2_Combinedmodel_trained_with_cnn_nn_trained_together_{number_lags}lags"
    model_dir = os.path.join(
        configuration.paths.model_dir,
        SITE,
        "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        configuration.paths.model_dir,
        SITE,
        "trained_models",
        f"member_{seed}_{model_name}_{SITE}_test_2.pth"
    )

    print(f"*** Loading model from: {model_path} ***")
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))

    # -----------------------------
    # 3. Run evaluation
    # -----------------------------
    combined_test_loader = datasets["test_loader"]

    reset_seeds(g, seed)
    y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
        CombinedModel=model,
        cnn=CNN_model_loaded,
        nn=NN_model,
        test_loader=combined_test_loader,
        print_accuracies=True,
        train_alone=False
    )

    # -----------------------------
    # 4. Save evaluation results
    # -----------------------------
    # seed_results = {
    #    'y_true_pred_pairs': (y_true, y_pred),
    #    'out_probs_seed': outputs_prob,
    #    'extreme_accuracy': extreme_acc,
    #    'nonextreme_accuracy': nonextreme_acc,
    #    'losses_train': losses_train_combined,
    #    'losses_val': losses_val_combined
    #}

    seed_results = {
        'y_true_pred_pairs': (y_true, y_pred),
        'out_probs_seed': outputs_prob,
        'extreme_accuracy': extreme_acc,
        'nonextreme_accuracy': nonextreme_acc,

    }

    results_file = os.path.join(
        configuration.paths.results_dir,
        SITE,
        f"{SITE}_{percentile}_{configuration.seed}",
        f"{SITE}_{percentile}_{configuration.seed}_evaluation.pkl"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "wb") as f:
        pickle.dump(seed_results, f)

    print(f"*** Results saved in {results_file} ***")

