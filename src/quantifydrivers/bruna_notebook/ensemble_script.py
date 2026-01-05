import os
import torch
import numpy as np
import pickle
import glob
from quantifydrivers import machine_learning
from quantifydrivers.machine_learning import convnext_functions


def generate_ensemble_seeds(fixed_seed=123, num_members=20):
    rng = np.random.default_rng(fixed_seed)
    seeds = rng.integers(low=0, high=2 ** 32 - 1, size=num_members).tolist()
    return seeds


def find_result_file(base_dir, site, seed, percentile, pattern_type="results"):
    """
    Helper to find the specific pickle file for a given seed.
    You might need to adjust the glob pattern depending on how exactly
    your filenames are saved in your experiments.
    """
    # EXAMPLE PATTERN: adapting to the structure seen in your refactored code
    # We assume you might have saved files like: .../cordoba_90p_results_SEED/results.pkl
    # Or strict searching within the site folder

    # Search recursively for a file that matches the criteria
    # This is a placeholder logic - you must adapt to your exact folder structure
    search_pattern = os.path.join(base_dir, site, f"*{seed}*", f"*{pattern_type}*.pkl")
    files = glob.glob(search_pattern)

    if not files:
        # Try alternative pattern from your 'Old' notebook
        search_pattern_old = os.path.join(base_dir, site, f"*{percentile}*{pattern_type}*{seed}*.pkl")
        files = glob.glob(search_pattern_old)

    if files:
        return files[0]  # Return the first match
    return None


def run_ensemble_analysis(configuration, datasets, device, timestamp):
    print(f"*** Starting Ensemble Analysis for Site: {configuration.site} ***")

    site = configuration.site
    percentile = configuration.percentile

    # 1. Setup Datasets (We only need Test for Evaluation)
    test_dataset = datasets["test_dataset"]
    test_loader = datasets["test_loader"]

    # 2. Define Seeds
    # You can define fixed_seed and num_members in your config.yaml
    ensemble_seed_config = getattr(configuration, "ensemble", {})
    fixed_seed = ensemble_seed_config.get("fixed_seed", 123)
    num_members = ensemble_seed_config.get("members", 20)
    list_seeds = generate_ensemble_seeds(fixed_seed, num_members)

    print(f"Aggregating over {len(list_seeds)} seeds...")

    # 3. Initialize Accumulators
    # We need the shapes from the dataset to initialize arrays
    num_samples = len(test_dataset)
    num_classes = 2  # Assuming binary

    # Shape: [Members, Samples, Classes]
    outputs_prob_seeds = np.zeros((len(list_seeds), num_samples, num_classes))

    # SHAP accumulators (Initialize lazily or using known shapes)
    ensemble_shap_nn = None
    ensemble_shap_cnn = None

    valid_seeds_count = 0

    # 4. Loop over seeds to load Pre-Computed Results
    for i, seed in enumerate(list_seeds):
        print(f"Processing Seed {seed} ({i + 1}/{len(list_seeds)})...")

        # A. Find and Load Evaluation Results (Predictions)
        res_file = find_result_file(configuration.paths.results_dir, site, seed, percentile, pattern_type="results")

        if not res_file or not os.path.exists(res_file):
            print(f"  [Warning] Results file not found for seed {seed}. Skipping.")
            continue

        with open(res_file, 'rb') as f:
            seed_results = pickle.load(f)

        # Store Probabilities
        # Assuming seed_results has key 'out_probs_seed' from your evaluation_script.py
        outputs_prob_seeds[i] = seed_results['out_probs_seed']

        # B. Find and Load SHAP Results
        shap_file = find_result_file(configuration.paths.results_dir, site, seed, percentile, pattern_type="SHAP")

        if shap_file and os.path.exists(shap_file):
            with open(shap_file, 'rb') as f:
                shap_data = pickle.load(f)

            # Initialize arrays on first valid load
            if ensemble_shap_nn is None:
                # Shape: [Members, Samples, Features]
                ensemble_shap_nn = np.zeros((len(list_seeds), *shap_data['nn'].shape))
                # Shape: [Members, Samples, V, H, W] (or similar)
                ensemble_shap_cnn = np.zeros((len(list_seeds), *shap_data['cnn'].shape))

            ensemble_shap_nn[i] = shap_data['nn']
            ensemble_shap_cnn[i] = shap_data['cnn']
        else:
            print(f"  [Warning] SHAP file not found for seed {seed}.")

        valid_seeds_count += 1

    if valid_seeds_count == 0:
        print("❌ No valid results found. Aborting ensemble analysis.")
        return

    # 5. Compute Statistics (Mean / Std)
    # Filter out zeros if some seeds were skipped (optional, depends on implementation)
    # Here we assume we want the mean of the valid ones

    output_prob_ensemble = np.mean(outputs_prob_seeds[:valid_seeds_count], axis=0)
    std_ensemble = np.std(outputs_prob_seeds[:valid_seeds_count], axis=0)

    shap_mean_nn = np.mean(ensemble_shap_nn[:valid_seeds_count], axis=0) if ensemble_shap_nn is not None else None
    shap_std_nn = np.std(ensemble_shap_nn[:valid_seeds_count], axis=0) if ensemble_shap_nn is not None else None

    shap_mean_cnn = np.mean(ensemble_shap_cnn[:valid_seeds_count], axis=0) if ensemble_shap_cnn is not None else None
    shap_std_cnn = np.std(ensemble_shap_cnn[:valid_seeds_count], axis=0) if ensemble_shap_cnn is not None else None

    # 6. Evaluate Ensemble Performance
    # We need a dummy model structure for the evaluation function,
    # even if we just pass the probabilities.

    # Re-instantiate architecture (Dummy weights, just for function signature)
    # Using your existing logic from evaluation_script.py
    train_dataset = datasets["train_dataset"]
    train_era5 = datasets["train_era5"]

    NN_dummy = machine_learning.ToCombineExtremeClassifier(
        input_dim=len(train_dataset.all_features), train_alone_NN=False, num_classes=2
    ).to(device)

    CNN_dummy = convnext_functions.ConvNext(
        num_channels=len(train_era5.all_features), num_classes=2,
        patch_size=4, layer_dims=[4, 6, 6, 16], depths=[1, 2, 2, 1],
        drop_rate=0.05, train_alone=False
    ).to(device)

    model_dummy = machine_learning.CombinedModel(
        NN_dummy, CNN_dummy, nn_hidden_dim=8, cnn_hidden_dim=16, output_dim=2
    ).to(device)

    print("*** Running Ensemble Evaluation ***")
    # Note: Ensure machine_learning.evaluate_ensamble accepts 'probs_ensamble'
    y_true, y_pred, extreme_acc, nonextreme_acc = machine_learning.evaluate_ensamble(
        CombinedModel=model_dummy,
        cnn=CNN_dummy,
        nn=NN_dummy,
        test_loader=test_loader,
        probs_ensamble=output_prob_ensemble,
        print_accuracies=True
    )

    # 7. Save Aggregate Results
    ensemble_results = {
        'y_true_pred_pairs': (y_true, y_pred),
        'out_probs_ensemble': output_prob_ensemble,
        'std_probs_ensemble': std_ensemble,
        'extreme_accuracy': extreme_acc,
        'nonextreme_accuracy': nonextreme_acc,
        'shap_nn_mean': shap_mean_nn,
        'shap_nn_std': shap_std_nn,
        'shap_cnn_mean': shap_mean_cnn,
        'shap_cnn_std': shap_std_cnn
    }

    out_file = os.path.join(
        configuration.paths.results_dir,
        site,
        f"{site}_{percentile}_ENSEMBLE_results_{timestamp}.pkl"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'wb') as f:
        pickle.dump(ensemble_results, f)

    print(f"*** Ensemble Results Saved: {out_file} ***")