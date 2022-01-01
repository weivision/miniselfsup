# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability


HEAD_REGISTRY = Registry('HEAD')


def build_head(cfg):
    avai_heads = HEAD_REGISTRY.registered_names()
    check_availability(cfg.name, avai_heads)
    print('| ------ head: {}'.format(cfg.name))
    
    return HEAD_REGISTRY.get(cfg.name)(cfg)
