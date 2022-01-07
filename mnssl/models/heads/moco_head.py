# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from mnssl.utils import concat_all_gather
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class MoCoHead(nn.Module):
    """
    Build a MoCo head.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCoHead, self).__init__()

        self.temperature = cfg.temperature
        self.queue_len = cfg.queue_len
        self.feat_dim = cfg.feat_dim

        # create the queue
        self.register_buffer('queue', torch.randn(cfg.feat_dim, cfg.queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q, k):

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)
        
        return dict(loss=loss)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr
