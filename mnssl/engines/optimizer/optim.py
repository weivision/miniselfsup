# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
from .build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
def SimSiam(cfg, model):

    if cfg.fix_head_lr:
        param_groups = [{'params': model.backbone.parameters(), 'fix_lr': False},
                        {'params': model.neck.parameters(), 'fix_lr': False},
                        {'params': model.head.parameters(), 'fix_lr': True}]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    return optimizer


@OPTIMIZER_REGISTRY.register()
def SwAV(cfg, model):

    optimizer = torch.optim.SGD(model.parameters(), cfg.base_lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    if cfg.lars:
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    return optimizer


@OPTIMIZER_REGISTRY.register()
def MoCo(cfg, model):

    optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    return optimizer


@OPTIMIZER_REGISTRY.register()
def BYOL(cfg, model):

    decay = list()
    no_decay = list()
    for name, m in model.named_parameters():
        if 'bn' in name or 'gn' in name or 'bias' in name:
            no_decay.append(m)
        else:
            decay.append(m)
    
    param_groups = [{'params': decay, 'decay': True},
                    {'params': no_decay, 'decay': False}]

    optimizer = torch.optim.SGD(param_groups, cfg.base_lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)

    for param_group in optimizer.param_groups:
        if 'decay' in param_group and param_group['decay']:
            param_group['weight_decay'] = cfg.weight_decay
        else:
            param_group['weight_decay'] = 0.
    
    if cfg.lars:
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    
    return optimizer


@OPTIMIZER_REGISTRY.register()
def LinearCls(cfg, model):

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.SGD(parameters, cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    
    if cfg.lars:
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    return optimizer
