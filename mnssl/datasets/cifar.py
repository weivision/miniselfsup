# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import errno
import io
import os
import pathlib
import warnings
from collections import defaultdict

import mc
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@DATASET_REGISTRY.register()
class CIFAR(Dataset):
    def __init__(self, root: str, train, transform, read_from="mc"):
        if not os.path.exists(root):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root)
