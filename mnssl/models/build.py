# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import check_availability

from .algorithms import ALGORITHM_REGISTRY, BYOL, LinearCls, MoCo, SimSiam, SwAV  # noqa


def build_model(cfg):

    avai_algos = ALGORITHM_REGISTRY.registered_names()
    check_availability(cfg.name, avai_algos)
    print("| --- algorithm: {}".format(cfg.name))

    return ALGORITHM_REGISTRY.get(cfg.name)(cfg)
