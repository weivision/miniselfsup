# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability


SCHEDULER_REGISTRY = Registry('SCHEDULER')


def build_scheduler(cfg, optimizer):
    avai_schedulers = SCHEDULER_REGISTRY.registered_names()
    check_availability(cfg.name, avai_schedulers)
    print('| --- scheduler: {}'.format(cfg.name))
    
    return SCHEDULER_REGISTRY.get(cfg.name)(
        cfg, optimizer
    )
