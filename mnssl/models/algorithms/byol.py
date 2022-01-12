# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import copy
from math import cos, pi

import torch
import torch.nn as nn

from ..backbones.build import build_backbone
from ..build import ALGORITHM_REGISTRY
from ..heads.build import build_head
from ..necks.build import build_neck
from .base import BaseMethod


@ALGORITHM_REGISTRY.register()
class BYOL(BaseMethod):
    """
    Build a BYOL model.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs.
        """
        super(BYOL, self).__init__()

        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)

        self.backbone_t = copy.deepcopy(self.backbone)
        self.neck_t = copy.deepcopy(self.neck)

        for param_o, param_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.neck.parameters(), self.neck_t.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        self.head = build_head(cfg.head)

        self.base_momentum = cfg.base_momentum
        self.end_momentum = cfg.end_momentum
        self.momentum = cfg.base_momentum
        self.update_interval = cfg.update_interval
        self.epoch = 0
        self.iteration = 0
        self.max_iter = 0

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_o, param_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_t in zip(self.neck.parameters(), self.neck_t.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def forward(self, inputs):

        assert isinstance(inputs, list)
        img_v1 = inputs[0].cuda(non_blocking=True)
        img_v2 = inputs[1].cuda(non_blocking=True)

        # compute online features
        online_z1 = self.neck(self.backbone(img_v1))  # NxC
        online_z2 = self.neck(self.backbone(img_v2))  # NxC

        # compute target features
        with torch.no_grad():
            target_z1 = self.neck_t(self.backbone_t(img_v1))  # NxC
            target_z2 = self.neck_t(self.backbone_t(img_v2))  # NxC

        loss = 2.0 * (
            self.head(online_z1, target_z2)["loss"] + self.head(online_z2, target_z1)["loss"]
        )
        return dict(loss=loss)

    def train_update(self, scheduler):
        self.max_iter = scheduler.epochs * scheduler.iter_per_epoch

    def iter_update(self):

        if (self.iteration + 1) % self.update_interval == 0:
            self.momentum = (
                self.end_momentum
                - (self.end_momentum - self.base_momentum)
                * (cos(pi * self.iteration / float(self.max_iter)) + 1)
                / 2
            )
            self.momentum_update()

        self.iteration += 1
