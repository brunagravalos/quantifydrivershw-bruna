__all__ = ["data_files", "functions", "hypm_tunning", "notebooks", "prepare_inputs_model","train_and_shap"]
from . import data_files, functions, hypm_tunning, notebooks, prepare_inputs_model, train_and_shap

import os
import logging

logger = logging.getLogger(__name__)

bsc_machine = os.environ.get("BSC_MACHINE", None)
if bsc_machine in ["mn5", "amd"]:
    _ESARCHIVE_ = "/gpfs/projects/bsc32/esarchive_cache/"
else:
    _ESARCHIVE_ = "/esarchive/"

