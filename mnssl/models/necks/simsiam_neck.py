# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn

from .build import NECK_REGISTRY


@NECK_REGISTRY.register()
class SimSiamNeck(nn.Module):
    """
    Build a SimSiam neck.
    """
    def __init__(self, cfg):
        """
        Args:

        cfg: configs
        """
        super(SimSiamNeck, self).__init__()

        # build a 3-layer encoder
        self.encoder = nn.Sequential(nn.Linear(cfg.input_dim, cfg.input_dim, bias=False), # first layer
                                     nn.BatchNorm1d(cfg.input_dim),
                                     nn.ReLU(inplace=True), 
                                     nn.Linear(cfg.input_dim, cfg.input_dim, bias=False), # second layer
                                     nn.BatchNorm1d(cfg.input_dim),
                                     nn.ReLU(inplace=True), 
                                     nn.Linear(cfg.input_dim, cfg.input_dim, bias=False), # output layer
                                     nn.BatchNorm1d(cfg.output_dim, affine=False)) 

    def forward(self, x):
        z = self.encoder(x) # NxC
        return z
