# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from mnssl.utils import Registry, check_availability
ALGORITHM_REGISTRY = Registry('ALGORITHM')

from .algorithms import SimSiam
from .algorithms import MoCo
from .algorithms import SwAV
from .algorithms import BYOL
from .algorithms import LinearCls


def build_model(cfg):

    avai_algos = ALGORITHM_REGISTRY.registered_names()
    check_availability(cfg.name, avai_algos)
    print('| --- algorithm: {}'.format(cfg.name))

    return ALGORITHM_REGISTRY.get(cfg.name)(cfg)
