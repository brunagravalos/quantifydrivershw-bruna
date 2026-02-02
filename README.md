## 1. Installation
### Setup Repository
The project repository is available in github at this link. To get started, clone the project and navigate to the directory:
```
git clone repository_url
cd repository_name
```

### Environment Setup

This project uses uv for dependency management.
1. **Install uv** (if not installed):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
_Note: Restart your shell to ensure uv is in your PATH._

2. **Create a Local Virtual Environment:**
```
[ -d ".venv" ] || uv venv
```
3. **Init and sync Environment:** 
```
uv init
uv sync
```
4. **Activate venv**
```bash
source .venv/bin/activate
```

## 2. Training, Evaluation & SHAP computing

The workflow for this project follows a pipeline designed to replicate the methodology from the associated paper. The process is divided into four distinct stages:
- **Data Loading:** Retrieves the local-scale (ERA5-Land) and large-scale (ERA5) datasets, applies preprocessing (standardization, lagging), and prepares the PyTorch DataLoaders.

- **Training:** Initializes the Hybrid Model (MLP + ConvNext), handles class imbalance with weighted loss, and trains the model using site-specific hyperparameters. The files with the fine-tuned hyperparameters are provided in the folder `data_files` (in the repository itself).

- **Evaluation:** Loads the trained model weights and computes performance metrics (extreme vs. non-extreme accuracy) on the test set.

- **SHAP Computing:** Uses shap.GradientExplainer to compute feature importance scores for both the local (NN) and large-scale (CNN) components.

There are two available scripts to run these processes. The first one runs the entire pipeline end-to-end, while the second is modular (assuming weights exist). Both scripts utilize Hydra, meaning you can modify their behavior via the command line without changing the source code.

#### 1. Full Pipeline (`training_evaluation_SHAP_pipeline.py`)

This script executes the entire lifecycle: it loads data, trains a new model from scratch, evaluates it, and computes SHAP values.

Example SLURM Script:
```
#!/bin/bash
#SBATCH --job-name=train_cordoba
#SBATCH --output=train_cordoba.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

SCRIPT_PATH="/path/to/training_evaluation_SHAP_pipeline.py"

# Run the pipeline with the defaults (the site will be Cordoba)
uv run python "$SCRIPT_PATH"
```
*Note: This will run using the defaults defined in conf/config.yaml. Specific file paths must be configured for your environment (see below).*

#### 2. Evaluation & SHAP Only (`evaluation_SHAP_pipeline.py`)
This script assumes that you have some model weights saved and only does the model evalutaion and SHAP values computing.


### Configuration

This project uses **Hydra** for configuration management. This strictly separates the code logic from experimental settings. You should not edit the Python code to change parameters; instead, use the configuration files or CLI overrides.

The main configuration entry point is `conf/config.yaml`. It is composed of several groups:

```
conf/
├── config.yaml
├── dataset/
├── paths/
├── site/
└── hyperparameters/
```

#### 1. Configuration groups

| Group | Description | Default |
| :--- | :--- | :--- |
| `site` | The geographical site to model. Supported sites: `cordoba`, `lyon`, `marrakech`, `belgrado`, `stockholm`, `hannover`. | `cordoba` |
| `dataset` | Defines variables, date ranges, and lag settings for ERA5 and ERA5-Land data. | `default` |  
| `paths` | Defines the locations of input NetCDF files and output directories. | `marenostrum` |
| `hyperparameters` | Specific training settings (learning rate, weight decay, minority_weight_multiplier) for the selected site. | `default` |



#### 2. Key Attributes

- `seed`: `(int)` The random seed for reproducibility.
- `percentile`: `(str)` The extreme event threshold (e.g., "90p", "95p"). This determines which label file is loaded.
- `epoch_config.epochs`: `(int)` Number of training epochs (default: 75).

#### 3. Detailed Group Configuration

**A. Site**

As mentioned before, at the moment the model does the training and evaluation for a specific site. The available sites are `cordoba`, `lyon`, `marrakech`, `belgrado`, `stockholm` and `hannover`.

**B. Dataset**

This group defines the physical variables and temporal structure of the data. Keep in mind that the default configuration has the values used for the paper.

- **Variables**:
    - `variables_era5`: List of large-scale atmospheric variables. Allowed values: `["g500", "g200", "psl"]`.
    - `variables_era5land`: List of local soil variables. Allowed values: `["swvl1", "swvl2", "swvl3"]`.

- **Time Range**:
    - `start_date_train` / `end_date_train`: Training period (e.g., `"1950-01-01"` to `"2013-12-31"`).
    - `start_date_test` / `end_date_test`: Testing period.
    - `months`: List of integers representing the season of interest (e.g., `[6, 7, 8]` for JJA).

- **Lagging Strategy**:
    - `start_lag`: (int) The starting lag for the ERA5 predictor variables.
    - `lags_era5`: (int) The number of lag steps to include.

- **Drought Indices (in developing)**:
    - `use_spei`: (bool) If True, includes SPEI/SPI indices in the model.
    - `spei_spi`: Type of index (`"spei"` or `"spi"`).
    - `scales_spei`: List of time scales (e.g., `['30', '60']`).

**C. Paths**

This group defines the absolute paths to input data and output directories. These must be valid locations on the filesystem (e.g., the cluster).

    - **Input Data Files (ERA5 & Auxiliaries)**
        - `file_spei` / `file_spi`: Paths to the directories containing Standardized Precipitation (Evapotranspiration) Index data. Only used if use_spei is set to True in the dataset config.

    - **Base Directory**
        - base_folder: The root directory for the project's data storage (pointing to the Zarr file, more information in the [Data section](./data.md)).

    - **Output Directories**
        - `model_dir`: Directory where trained model weights (.pth files) are saved.
        - `results_dir`: Directory where evaluation pickles (metrics, loss history, predictions) and computed SHAP feature files are saved.

**D. Hyperparameters**

These control the training dynamics. If `default_hypms` is `True`, the system loads optimized parameters from `site_hypms`:
    - `lr`: Learning rate (range: `1e-5` to `1e-3`).
    - `w_decay`: Weight decay for regularization (range: `1e-5` to `1e-1`).
    - `minority_weight_multiplier`: Multiplier to handle class imbalance (1.0 to 10.0).
If `default_hypms` is `False`, the data loading script will use the optimized values from the folder `data_files/HYPMS_optimization_results` included in the project source code.

### How to use

#### Changing Values via CLI

Hydra allows you to override any config value directly from the command line using dot notation.

Run for a different `site` and `percentile`:
```
python training_evaluation_SHAP_pipeline.py site=stockholm percentile="95p"
```
Run with a specific `seed` and custom `learning rate`:
```
python training_evaluation_SHAP_pipeline.py seed=42 hyperparameters.site_hypms.lr=0.0005
```
#### Creating a New Default

If you are moving to a new cluster and need to change the path configuration, or if you are running the code for the first time you can do the following:

    1. Create a new file: `conf/paths/my_cluster.yaml`.

    2. Fill in the required paths (base folder, model dir, etc.).

    3. Run with: ` uv run python training_evaluation_SHAP_pipeline.py paths=my_cluster`.

#### Seeds

To reproduce the paper's ensemble results, you should run the pipeline multiple times with different seeds. The paper uses a standard ensemble of 20 seeds to plot the final results (e.g., generated from a master seed or a fixed list like given below). Hydra allows multirun calls with the -m flag, so you can run the pipeline with `uv run python training_evaluation_SHAP_pipeline.py -m seed=1234, ....`. You could also parallelize with the different seeds.
Here is an example SLURM script that parallelizes with the list of 20 seeds used in the paper:
```
#SBATCH --array=1-20

#list of seeds
SEED_LIST=(66316748 2930678936 2546691362 231159514 3904498325 946438445 1095601156 791870896 1432871125 755510091 1493800520 3487919346 1938714511 3965736568 1930440936 1187877992 3387705611 3520819031 3701866991 3822060012)

CURRENT_SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID - 1]}

uv run python "$SCRIPT_PATH" seed=$CURRENT_SEED site=marrakech

```

## 3. Visualization & Analysis

To reproduce the figures from the paper, the project provides two notebooks. These are designed to aggregate the results from the multi-seed ensemble and generate plots.

To run the notebooks, ensure you have the project installed as a kernel:

```
# Install the kernel
uv run python -m ipykernel install --user --name quantifydrivers --display-name "Python (quantifydrivers)"

# Install the project in editable mode
uv pip install -e .
```
*Select the **Python (quantifydrivers)** kernel when opening the notebooks.*

### A. Ensemble Aggregation (`ensemble_processing.ipynb`)
Since the model is trained on 20 different seeds to ensure robustness, this notebook acts as the aggregator. It iterates through the results of all seeds defined in your configuration and performs two key tasks:

    - **Metric Aggregation:** It computes the mean and standard deviation for performance metrics (Accuracy, F1-Score, Precision, Recall) and SHAP values across the entire ensemble.

    - **Training Stability Plot:** It generates the Loss History plot, visualizing the mean training and validation loss curves (with standard deviation shading) to verify model convergence and check for overfitting.

Output: Saves a processed dictionary containing the aggregated metrics and averaged SHAP values for the next step.
### B. Final Figures (`main_plots.ipynb`)

This notebook loads the aggregated data produced by the previous step and generates the primary figures used to evaluate the scientific validity of the model:

    - **Model Performance (ROC Curves):** Plots the Receiver Operating Characteristic (ROC) curve with the Area Under the Curve (AUC) score. It visualizes the trade-off between the True Positive Rate and False Positive Rate, showing the mean curve of the ensemble with confidence intervals.

    - **Driver Importance (SHAP):** Visualizes the Global Feature Importance. It produces bar charts ranking the top atmospheric (ERA5) and land (ERA5-Land) drivers based on their mean absolute SHAP values, quantifying which physical variables most influence the heatwave predictions.
