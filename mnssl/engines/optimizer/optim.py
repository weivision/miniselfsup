# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
from .build import OPTIMIZER_REGISTRY


class BaseOptimizer:

    def __init__(self, cfg=None, model=None):
        pass
    
    def update(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict=None):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()


@OPTIMIZER_REGISTRY.register()
class SimSiam(BaseOptimizer):

    def __init__(self, cfg=None, model=None):
        super(SimSiam, self).__init__()

        self.fix_head_lr = cfg.fix_head_lr

        if self.fix_head_lr:
            self.param_groups = [{'params': model.backbone.parameters(), 'fix_lr': False},
                                 {'params': model.neck.parameters(), 'fix_lr': False},
                                 {'params': model.head.parameters(), 'fix_lr': True}]
        else:
            self.param_groups = model.parameters()

        self.optimizer = torch.optim.SGD(self.param_groups, cfg.lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)


@OPTIMIZER_REGISTRY.register()
class SwAV(BaseOptimizer):

    def __init__(self, cfg=None, model=None):
        super(SwAV, self).__init__()

        self.optimizer = torch.optim.SGD(model.parameters(), cfg.base_lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)

        if cfg.lars:
            from apex.parallel.LARC import LARC
            self.optimizer = LARC(optimizer=self.optimizer, trust_coefficient=0.001, clip=False)

        self.param_groups = self.optimizer.param_groups


@OPTIMIZER_REGISTRY.register()
class MoCo(BaseOptimizer):

    def __init__(self, cfg=None, model=None):
        super(MoCo, self).__init__()

        self.optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        
        self.param_groups = self.optimizer.param_groups


@OPTIMIZER_REGISTRY.register()
class BYOL(BaseOptimizer):

    def __init__(self, cfg=None, model=None):
        super(BYOL, self).__init__()

        decay = list()
        no_decay = list()
        for name, m in model.named_parameters():
            if 'bn' in name or 'gn' in name or 'bias' in name:
                no_decay.append(m)
            else:
                decay.append(m)
        
        param_groups = [{'params': decay, 'decay': True},
                        {'params': no_decay, 'decay': False}]

        self.optimizer = torch.optim.SGD(param_groups, cfg.base_lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)

        for param_group in self.optimizer.param_groups:
            if 'decay' in param_group and param_group['decay']:
                param_group['weight_decay'] = cfg.weight_decay
            else:
                param_group['weight_decay'] = 0.
        
        if cfg.lars:
            from apex.parallel.LARC import LARC
            self.optimizer = LARC(optimizer=self.optimizer, trust_coefficient=0.001, clip=False)

        self.param_groups = self.optimizer.param_groups


@OPTIMIZER_REGISTRY.register()
class LinearCls(BaseOptimizer):

    def __init__(self, cfg=None, model=None):
        super(LinearCls, self).__init__()

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

        self.optimizer = torch.optim.SGD(parameters, cfg.lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        
        if cfg.lars:
            from apex.parallel.LARC import LARC
            self.optimizer = LARC(optimizer=self.optimizer, trust_coefficient=.001, clip=False)

        self.param_groups = self.optimizer.param_groups
