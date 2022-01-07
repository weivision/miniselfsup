# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class ClsHead(nn.Module):
    """
    Build a Classification head.
    """
    def __init__(self, cfg):
        """
        Args:

        cfg: configs
        """
        super(ClsHead, self).__init__()

        # build a 1-layer predictor
        self.predictor = nn.Linear(cfg.input_dim, cfg.num_classes)

        self.predictor.weight.data.normal_(mean=0.0, std=0.01)
        self.predictor.bias.data.zero_()

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z, labels):
        
        p = self.predictor(z) # NxC
        loss = self.criterion(p, labels)
        return dict(pred=p, loss=loss)
