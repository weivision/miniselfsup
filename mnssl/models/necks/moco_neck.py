# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch.nn as nn

from .build import NECK_REGISTRY


@NECK_REGISTRY.register()
class MoCoNeck(nn.Module):
    """
    Build a MoCo neck.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCoNeck, self).__init__()

        # build a 2-layer encoder
        self.mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hid_dim), nn.ReLU(), nn.Linear(cfg.hid_dim, cfg.output_dim)
        )

    def forward(self, x):
        z = self.mlp(x)  # NxC
        return z
