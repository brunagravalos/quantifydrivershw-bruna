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



def evaluation(config_path,datasets, seed, generator, losses_train_combined, losses_val_combined, model,CNN_model_loaded, NN_model):
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


    combined_test_loader = datasets["test_loader"]


    reset_seeds(seed)
    y_true, y_pred, outputs_prob, extreme_acc, nonextreme_acc = machine_learning.evaluate_CombinedModel(
        CombinedModel=model, cnn=CNN_model_loaded, nn=NN_model, test_loader=combined_test_loader,
        print_accuracies=True, train_alone=False)
    reset_seeds(seed)
    print(
        f"*** Evaluation complete. Extreme Accuracy: {extreme_acc:.4f}, Non-Extreme Accuracy: {nonextreme_acc:.4f} ***")  # NEW PRINT

    # Save the dictionaries with the relevant data ---------------------------------------------------------------------
    seed_results = {
        'y_true_pred_pairs': (y_true, y_pred),  # These are from the current site
        'out_probs_seed': outputs_prob,
        'extreme_accuracy': extreme_acc,
        'nonextreme_accuracy': nonextreme_acc,
        'losses_train': losses_train_combined,
        'losses_val': losses_val_combined
    }

    main_path = '/gpfs/scratch/bsc32/bsc214253/results/'  # Change to your desired path
    results_file_path = os.path.join(main_path, f'{CONF["SITE"]}/{percentile_to_load}_results_data_{seed}.pkl')

    results_dir = os.path.dirname(results_file_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    with open(results_file_path, 'wb') as f:
        pickle.dump(seed_results, f)
        print(f"saved file results {seed}")  # ORIGINAL PRINT
        print(f"*** Results dictionary saved to: {results_file_path} ***")  # NEW PRINT

    print("Finished training model, computing SHAP")

    return model, CNN_model_loaded, NN_model, losses_train_combined, losses_val_combined





