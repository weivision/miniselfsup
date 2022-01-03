# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch.nn as nn

from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..necks.build import build_neck


class BaseMethod(nn.Module):
    """Base algorithm.

    """
    def __init__(self, cfg=None):
        """
        Args:
            cfg: configs.
        """
        super(BaseMethod, self).__init__()
        pass

    def forward(self):
        pass

    def train_update(self, scheduler):
        pass
        
    def epoch_update(self):
        pass
    
    def iter_update(self):
        pass

    def optim_update(self):
        pass
