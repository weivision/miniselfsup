# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability


BACKBONE_REGISTRY = Registry('BACKBONE')


def build_backbone(cfg):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    check_availability(cfg.name, avai_backbones)
    print('| ------ backbone: {}'.format(cfg.name))
    
    return BACKBONE_REGISTRY.get(cfg.name)(
        eval_mode=cfg.eval_mode,
        zero_init_residual=cfg.zero_init_residual,
        padding_mode=cfg.padding_mode,
    )
