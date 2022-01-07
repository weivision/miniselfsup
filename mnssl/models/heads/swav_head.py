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
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class SwAVHead(nn.Module):
    """
    Build a SwAV head.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(SwAVHead, self).__init__()
        
        self.crops_for_assign = cfg.crops_for_assign
        self.num_crops = cfg.num_crops
        self.epsilon = cfg.epsilon
        self.temperature = cfg.temperature
        self.sinkhorn_iterations = cfg.sinkhorn_iterations
        
        # create the queue
        self.world_size = dist.get_world_size()
        self.queue_length = cfg.queue_length
        self.register_buffer("queue", torch.zeros(len(self.crops_for_assign), 
                             self.queue_length // self.world_size, cfg.output_dim))
        
        self.epoch_queue_starts = cfg.epoch_queue_starts
        self.epoch = 0

    def forward(self, embeddings, outputs, protos):
        
        bs = int(embeddings.size(0)/sum(self.num_crops))
        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = outputs[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if self.epoch >= self.epoch_queue_starts:
                    if not torch.all(self.queue[i, -1, :] == 0):
                        out = torch.cat((torch.mm(self.queue[i], protos), out))

                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embeddings[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = self.distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                x = outputs[bs * v: bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.num_crops) - 1)
        loss /= len(self.crops_for_assign)

        return dict(loss=loss)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.world_size # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
