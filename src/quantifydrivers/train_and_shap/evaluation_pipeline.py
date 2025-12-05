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
    print("*** Starting Evaluation Phase on Test Data ***")  # NEW PRINT

    with open(config_path, "r") as f:
        CONF = yaml.safe_load(f)

    g = generator
    # ==============================================================================================================
    # Choose which percentile's hyperparameters to load
    percentile_to_load = '90p'

    combined_test_loader = datasets["test_loader"]
    reset_seeds(g,seed)
    y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
        CombinedModel=model, cnn=CNN_model_loaded, nn=NN_model, test_loader=combined_test_loader,
        print_accuracies=True, train_alone=False)
    reset_seeds(g,seed)
    print(f"*** Evaluation complete. Extreme Accuracy: {extreme_acc:.4f}, Non-Extreme Accuracy: {nonextreme_acc:.4f} ***")  # NEW PRINT

    # Save the dictionaries with the relevant data ---------------------------------------------------------------------
    seed_results = {
        'y_true_pred_pairs': (y_true, y_pred),  # These are from the current site
        'out_probs_seed': outputs_prob,
        'extreme_accuracy': extreme_acc,
        'nonextreme_accuracy': nonextreme_acc,
        'losses_train': losses_train_combined,
        'losses_val': losses_val_combined
    }

    # Save results → YAML
    results_dir = CONF["paths"]["results_dir"]
    results_file = os.path.join(
        results_dir,
        CONF["SITE"],
        f"{CONF["percentile_to_load"]}_results_data_{seed}.pkl"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)


    with open(results_file, 'wb') as f:
        pickle.dump(seed_results, f)
        print(f"saved file results {seed}")  # ORIGINAL PRINT
        print(f"*** Results dictionary saved to: {results_file} ***")  # NEW PRINT

    print("Finished training model, computing SHAP")

    return





