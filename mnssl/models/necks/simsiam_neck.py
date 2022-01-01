# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
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
        
        input_dim=cfg.input_dim
        output_dim=cfg.output_dim

        # build a 3-layer encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim, bias=False),
                                     nn.BatchNorm1d(input_dim),
                                     nn.ReLU(inplace=True), # first layer
                                     nn.Linear(input_dim, input_dim, bias=False),
                                     nn.BatchNorm1d(input_dim),
                                     nn.ReLU(inplace=True), # second layer
                                     nn.Linear(input_dim, input_dim),
                                     nn.BatchNorm1d(output_dim, affine=False)) # output layer
        
        self.encoder[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
    
    def forward(self, x):
        z = self.encoder(x) # NxC
        return z
