# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import mc
import io
import os
import pathlib
import warnings
import errno
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np


@DATASET_REGISTRY.register()
class CIFAR(Dataset):

    def __init__(self, root: str, train, transform, read_from='mc'):
        if not os.path.exists(root):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), root)

