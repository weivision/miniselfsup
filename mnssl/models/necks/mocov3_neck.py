# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch.nn as nn

from .build import NECK_REGISTRY


@NECK_REGISTRY.register()
class MoCov3Neck(nn.Module):
    """
    Build a MoCov3 neck.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCov3Neck, self).__init__()

        # Sequential(
        #   (0): Linear(in_features=768, out_features=4096, bias=False)
        #   (1): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (2): ReLU(inplace=True)
        #   (3): Linear(in_features=4096, out_features=4096, bias=False)
        #   (4): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (5): ReLU(inplace=True)
        #   (6): Linear(in_features=4096, out_features=256, bias=False)
        #   (7): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        # )
        
        # build a 3-layer encoder
        self.mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False), 
            nn.BatchNorm1d(cfg.hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False),
            nn.BatchNorm1d(cfg.hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False),
            nn.BatchNorm1d(cfg.output_dim, affine=False), 
        )

    def forward(self, x):
        z = self.mlp(x)  # NxC
        return z
