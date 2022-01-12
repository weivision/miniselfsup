# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")


def build_optimizer(cfg, model):
    avai_optimizers = OPTIMIZER_REGISTRY.registered_names()
    check_availability(cfg.name, avai_optimizers)
    print("| --- optimizer: {}".format(cfg.name))

    return OPTIMIZER_REGISTRY.get(cfg.name)(cfg, model)
