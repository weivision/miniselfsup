# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
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
class SwAV(BaseMethod):
    """
    Build a SwAV model.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs.
        """
        super(SwAV, self).__init__()

        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)
        self.head = build_head(cfg.head)
        self.epoch = 0
        self.iteration = 0
    
    def forward(self, inputs):
        
        # normalize the prototypes
        with torch.no_grad():
            w = self.neck.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.neck.prototypes.weight.copy_(w)

        embeddings, outputs = self._forward_backbone_and_neck(inputs)
        embeddings = embeddings.detach()
        protos = self.neck.prototypes.weight.t()
        
        # self.head.epoch = self.epoch
        loss = self.head(embeddings, outputs, protos)['loss']
        return dict(loss=loss)
    
    def _forward_backbone_and_neck(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.neck(output)

    def epoch_update(self):
        self.epoch += 1
        self.head.epoch = self.epoch
    
    def iter_update(self):
        if self.iteration <= self.neck.freeze_prototypes_niters:
            self.iteration += 1
    
    def optim_update(self):
        # cancel gradients for the prototypes
        if self.iteration < self.neck.freeze_prototypes_niters:
            for name, p in self.neck.named_parameters():
                if "prototypes" in name:
                    p.grad = None
