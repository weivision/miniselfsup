# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class BYOLHead(nn.Module):
    """
    Build a BYOL head.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(BYOLHead, self).__init__()
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True), # hidden layer
                                       nn.BatchNorm1d(cfg.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False)) # output layer
    
    def forward(self, latent, target):
        
        pred = self.predictor(latent)
        loss = self.criterion(pred, target.detach())

        return dict(loss=loss)

    def criterion(self, pred, target):

        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -(pred_norm * target_norm).sum(dim=1).mean()

        return loss
