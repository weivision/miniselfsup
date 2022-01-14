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
class MoCo(BaseMethod):
    """
    Build a MoCo model.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCo, self).__init__()

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

        # compute query features
        q = self.neck(self.backbone(img_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.neck_k(self.backbone_k(img_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

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

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
