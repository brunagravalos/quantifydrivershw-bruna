__all__ = ["machine_learning"]
from . import machine_learning

import os
import logging

logger = logging.getLogger(__name__)

bsc_machine = os.environ.get("BSC_MACHINE", None)
if bsc_machine in ["mn5", "amd"]:
    _ESARCHIVE_ = "/gpfs/projects/bsc32/esarchive_cache/"
else:
    _ESARCHIVE_ = "/esarchive/"

