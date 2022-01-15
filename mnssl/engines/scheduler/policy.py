# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import math
import torch

import numpy as np

from .build import SCHEDULER_REGISTRY


class BaseScheduler:
    def __init__(self, cfg=None, optimizer=None):
        pass

    def epoch_step(self, epoch):
        pass

    def iter_step(self, epoch, iter):
        pass


@SCHEDULER_REGISTRY.register()
class SimSiam(BaseScheduler):
    def __init__(self, cfg=None, optimizer=None):
        super(SimSiam, self).__init__()

        self.optimizer = optimizer
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.iter_per_epoch = cfg.iter_per_epoch

    def epoch_step(self, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
        for param_group in self.optimizer.param_groups:
            if "fix_lr" in param_group and param_group["fix_lr"]:
                param_group["lr"] = self.lr
            else:
                param_group["lr"] = cur_lr


@SCHEDULER_REGISTRY.register()
class SwAV(BaseScheduler):
    def __init__(self, cfg=None, optimizer=None):
        super(SwAV, self).__init__()

        self.optimizer = optimizer
        self.epochs = cfg.epochs
        self.iter_per_epoch = cfg.iter_per_epoch

        warmup_lr_schedule = np.linspace(
            cfg.start_warmup, cfg.base_lr, cfg.iter_per_epoch * cfg.warmup_epochs
        )
        iters = np.arange(cfg.iter_per_epoch * (cfg.epochs - cfg.warmup_epochs))
        cosine_lr_schedule = np.array(
            [
                cfg.final_lr
                + 0.5
                * (cfg.base_lr - cfg.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi * t / (cfg.iter_per_epoch * (cfg.epochs - cfg.warmup_epochs))
                    )
                )
                for t in iters
            ]
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def iter_step(self, epoch, it):
        """Decay the learning rate based on schedule"""
        iteration = epoch * self.iter_per_epoch + it
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[iteration]


@SCHEDULER_REGISTRY.register()
class MoCo(BaseScheduler):
    def __init__(self, cfg=None, optimizer=None):
        super(MoCo, self).__init__()

        self.optimizer = optimizer
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.iter_per_epoch = cfg.iter_per_epoch

    def epoch_step(self, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr


@SCHEDULER_REGISTRY.register()
class LinearCls(BaseScheduler):
    def __init__(self, cfg=None, optimizer=None):
        super(LinearCls, self).__init__()

        self.optimizer = optimizer
        self.lr = cfg.lr
        self.epochs = cfg.epochs

    def epoch_step(self, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr


@SCHEDULER_REGISTRY.register()
class LinearSwAV(BaseScheduler):
    def __init__(self, cfg=None, optimizer=None):
        super(LinearSwAV, self).__init__()

        self.optimizer = optimizer
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.epochs, eta_min=cfg.final_lr
        )

    def epoch_step(self, epoch):
        """Decay the learning rate based on schedule"""
        self.scheduler.step()
