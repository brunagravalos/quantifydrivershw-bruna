import logging

logger = logging.getLogger(__name__)
# if pytorch is installed, import the pytorch trainer and data loader

import os

bsc_machine = os.environ.get('BSC_MACHINE', None)
if bsc_machine in ["mn5", "amd"]:
    _ESARCHIVE_ = "/gpfs/projects/bsc32/esarchive_cache/"
else:
    _ESARCHIVE_ = "/esarchive/"


from . import loaders, tools, verification, plots, scalers
