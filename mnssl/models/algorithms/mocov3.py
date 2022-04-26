# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import copy

import torch
import torch.nn as nn

from mnssl.utils import concat_all_gather

from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..necks.build import build_neck
from .base import ALGORITHM_REGISTRY, BaseMethod


@ALGORITHM_REGISTRY.register()
class MoCov3(BaseMethod):
    """
    Build a MoCov3 model.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCov3, self).__init__()

        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)

        self.backbone_k = copy.deepcopy(self.backbone)
        self.neck_k = copy.deepcopy(self.neck)

        self.head = build_head(cfg.head)

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.neck.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.momentum = cfg.momentum

    def forward(self, inputs):

        img_q = inputs[0].cuda(non_blocking=True)
        img_k = inputs[1].cuda(non_blocking=True)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        loss = self.head(q, k)["loss"]
        return dict(loss=loss)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        """Momentum update of the key backbone."""
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        """Momentum update of the key neck."""
        for param_q, param_k in zip(self.neck.parameters(), self.neck_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
