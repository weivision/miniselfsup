# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch.nn as nn

from .build import HEAD_REGISTRY


def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


@HEAD_REGISTRY.register()
class MoCov3Head(nn.Module):
    """
    Build a MoCov3 head.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: configs
        """
        super(MoCov3Head, self).__init__()

        # Sequential(
        # (0): Linear(in_features=256, out_features=4096, bias=False)
        # (1): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (2): ReLU(inplace=True)
        # (3): Linear(in_features=4096, out_features=256, bias=False)
        # (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        # )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False),  # hidden layer
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False), # output layer
            nn.BatchNorm1d(cfg.output_dim, affine=False),
        )

    def forward(self, latent, target):

        pred = self.predictor(latent)
        loss = self.criterion(pred, target.detach())

        return dict(loss=loss)

    def criterion(self, pred, target):

        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -(pred_norm * target_norm).sum(dim=1).mean()

        return loss
