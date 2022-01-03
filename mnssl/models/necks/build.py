# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability


NECK_REGISTRY = Registry('NECK')


def build_neck(cfg):
    avai_necks = NECK_REGISTRY.registered_names()
    check_availability(cfg.name, avai_necks)
    print('| ------ neck: {}'.format(cfg.name))
    
    return NECK_REGISTRY.get(cfg.name)(cfg)
