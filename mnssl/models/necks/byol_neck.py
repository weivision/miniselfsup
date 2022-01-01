# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn

from .build import NECK_REGISTRY


@NECK_REGISTRY.register()
class BYOLNeck(nn.Module):
    """
    Build a BYOL neck.
    """
    def __init__(self, cfg):
        """
        Args:

        cfg: configs
        """
        super(BYOLNeck, self).__init__()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True), # hidden layer
                                       nn.BatchNorm1d(cfg.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False)) # output layer
    
    def forward(self, x):
        z = self.projector(x) # NxC
        return z
