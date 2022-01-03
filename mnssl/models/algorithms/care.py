# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn

from ..build import ALGORITHM_REGISTRY
from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..necks.build import build_neck
from .base import BaseMethod


@ALGORITHM_REGISTRY.register()
class CARE(BaseMethod):
    """
    Build a CARE model.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs.
        """
        super(CARE, self).__init__()

        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)
        self.head = build_head(cfg.head)
    
    def forward(self, inputs):
        
        assert isinstance(inputs, list)
        img_v1 = inputs[0].cuda(non_blocking=True)
        img_v2 = inputs[1].cuda(non_blocking=True)
        
        z1 = self.neck(self.backbone(img_v1))  # NxC
        z2 = self.neck(self.backbone(img_v2))  # NxC
        
        loss = self.head(z1, z2)['loss']
        return dict(loss=loss)
