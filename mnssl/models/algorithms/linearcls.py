# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn

from ..build import ALGORITHM_REGISTRY
from ..backbones.build import build_backbone
from ..necks.build import build_neck
from ..heads.build import build_head
from .base import BaseMethod


@ALGORITHM_REGISTRY.register()
class LinearCls(BaseMethod):
    """
    Build a LinearCls model.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs.
        """
        super(LinearCls, self).__init__()

        self.backbone = build_backbone(cfg.backbone)

        # freeze all layers in backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        self.head = build_head(cfg.head)
    
    def forward(self, img, labels):
        
        z = self.backbone(img)  # NxC
        output = self.head(z, labels)
        return output
