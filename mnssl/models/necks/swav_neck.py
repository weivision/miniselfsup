# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn

from .build import NECK_REGISTRY


@NECK_REGISTRY.register()
class SwAVNeck(nn.Module):
    """
    Build a SwAV neck.
    """
    def __init__(self, cfg):
        """
        Args:

        cfg: configs
        """
        super(SwAVNeck, self).__init__()

        self.norm = cfg.normalize

        self.freeze_prototypes_niters = cfg.freeze_prototypes_niters
        
        # build a 2-layer projector
        self.projector = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.BatchNorm1d(cfg.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.hidden_dim, cfg.output_dim),
            )
        
        # prototypes
        self.prototypes = nn.Linear(cfg.output_dim, cfg.nmb_prototypes, bias=False)

    def forward(self, x):
        if self.projector is not None:
            x = self.projector(x)

        if self.norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x
