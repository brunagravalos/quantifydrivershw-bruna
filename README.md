
## 1. Installation
### 1.1 Setup Repository
Clone the project and navigate to the directory:
```
git clone repository_url
cd repository_name
```

### 1.2 Environment Setup

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
3. **Sync Environment:** 
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

- **Training:** Initializes the Hybrid Model (MLP + ConvNext), handles class imbalance with weighted loss, and trains the model using site-specific hyperparameters.

- **Evaluation:** Loads the trained model weights and computes performance metrics (extreme vs. non-extreme accuracy) on the test set.

- **SHAP Computing:** Uses shap.GradientExplainer to compute feature importance scores for both the local (NN) and large-scale (CNN) components.

We have two available scripts to do this. The first one (name.py) does all of this process. The second one assumes you have the model weights computed and does only the last steps. How to run, give an example slurm. exaplin this will run on all defaults, which is explained below. emphasis on that specific paths will have to be configured or wont work.

This model (you can read the whole paper here) works locally for a certain list of sites: cordoba, lyon, marrakech, belgrado, stockholm and hannover, and does the whole pipeline separately for each of those sites. To configure for which site you want to run the model and other specifics, as the path in which you have saved your data, the code uses hydra configurations. They allow you to whitch between defaults from CLI, and to create your own configuration files. Below is a little guide on how the configuration is structured and how to use. Explain the code shouldnt be changed, only config, in order to obtain the diff results options.

There are two primary ways to interact with this pipeline. Both scripts utilize Hydra, meaning you can modify their behavior via the command line without changing the source code.

1. Full Pipeline (`training_evaluation_SHAP_pipeline.py`)

This script executes the entire lifecycle: it loads data, trains a new model from scratch, evaluates it, and computes SHAP values.

Example SLURM Script:
```

#!/bin/bash
#SBATCH --job-name=train_cordoba
#SBATCH --output=train_cordoba.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

SCRIPT_PATH="/path/to/training_evaluation_SHAP_pipeline.py"

# Run the pipeline for Cordoba with defaults
uv run python "$SCRIPT_PATH" site=cordoba
```
*Note: This will run using the defaults defined in conf/config.yaml. Specific file paths must be configured for your environment (see below).*

2. Evaluation & SHAP Only (`evaluation_SHAP_pipeline.py`)
This script assumes that you have some model weights saved and only does the model evalutaion and SHAP values computing.


### Configuration

This project uses **Hydra** for configuration management. This strictly separates the code logic from experimental settings. You should not edit the Python code to change parameters; instead, use the configuration files or CLI overrides.

The main configuration entry point is `conf/config.yaml`. It is composed of several groups:

Explanation of all atributes. conf.yaml has this certains parts, with this defaults:

explain hyperparameters, paths, dataset site, seed, percentile. Then dwell more concretely for example in which are the paths that need to be provided. also specify which are the changes that are expected and which would be outside of the reproducibility scope.

#### 1. Configuration groups

| Group | Description | Default |
| :--- | :--- | :--- |
| `site` | The geographical site to model. Supported sites: `cordoba`, `lyon`, `marrakech`, `belgrado`, `stockholm`, `hannover`. | `cordoba` |
| `dataset` | Defines variables, date ranges, and lag settings for ERA5 and ERA5-Land data. | `default` |  
| `paths` | Crucial: Defines the locations of input NetCDF files and output directories. | `marenostrum` |
| `hyperparameters` | Specific training settings (learning rate, weight decay, batch size) for the selected site. | `default` |

#### 2. Key Attributes

- `seed`: (int) The random seed for reproducibility.
- `percentile`: (str) The extreme event threshold (e.g., "90p", "95p"). This determines which label file is loaded.
- `epoch_config.epochs`: (int) Number of training epochs (default: 75).

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

If you are moving to a new cluster and need to change the path configuration, or if you are running the code for the first time you can do the followfing:

    1. Create a new file: `conf/paths/my_cluster.yaml`.

    2. Fill in the required paths (base folder, model dir, etc.).

    3. Run with: ` uv run python training_evaluation_SHAP_pipeline.py paths=my_cluster`.

#### Seeds

To reproduce the paper's ensemble results, you should run the pipeline multiple times with different seeds. The paper uses a standard ensemble of 20 seeds to plot the final results (e.g., generated from a master seed or a fixed list like given below). Hydra allows multirun calls with the -m flag, so you can run the pipeline with `uv run python training_evaluation_SHAP_pipeline.py -m seed=1234, ....`.
The list of 20 seeds used in the paper is the following:
```
ensemble_seeds = 66316748,2930678936,2546691362,231159514,3904498325,946438445,1095601156,
791870896,1432871125,755510091,1493800520,3487919346,1938714511,3965736568,1930440936,
1187877992,3387705611,3520819031,3701866991,3822060012
```

#### Path Configuration Guide

The `paths.yaml` file (located in conf/paths/) is the central control room for input data and output locations. The pipeline uses Hydra to load these paths into the configuration.paths object.

You must ensure these paths point to the correct locations on your machine or cluster (e.g., MareNostrum).
DEPENDING ON WHAT WE DECIDE.

## 3. Plots

Explain the notebooks.
