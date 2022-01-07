# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class SimSiamHead(nn.Module):
    """
    Build a SimSiam head.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(SimSiamHead, self).__init__()
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False), # hidden layer
                                       nn.BatchNorm1d(cfg.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(cfg.hidden_dim, cfg.output_dim)) # output layer
        
        self.criterion = nn.CosineSimilarity(dim=1)
    
    def forward(self, z1, z2):
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + \
                 self.criterion(p2, z1.detach()).mean()) * 0.5
        return dict(loss=loss)
