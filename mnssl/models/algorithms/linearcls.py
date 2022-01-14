# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from ..backbones.build import build_backbone
from ..heads.build import build_head
from .base import ALGORITHM_REGISTRY, BaseMethod


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
