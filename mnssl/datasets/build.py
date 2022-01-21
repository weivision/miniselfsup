# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler
from mnssl.utils import Registry, check_availability

TRANSFORM_REGISTRY = Registry("TRANSFORM")
DATASET_REGISTRY = Registry("DATASET")


def build_transform(dataset, cfg):
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    check_availability(cfg.name, avai_transforms)
    transform = TRANSFORM_REGISTRY.get(cfg.name)(dataset, cfg)
    return transform


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.dataset, avai_datasets)
    print("| --- dataset: {}".format(cfg.dataset))

    transform = build_transform(cfg.dataset, cfg.train_transform)
    train_dataset = DATASET_REGISTRY.get(cfg.dataset)(
        root=cfg.root, train=True, transform=transform
    )

    if cfg.val_transform:
        transform = build_transform(cfg.dataset, cfg.val_transform)
        val_dataset = DATASET_REGISTRY.get(cfg.dataset)(
            root=cfg.root, train=False, transform=transform
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset


def build_dataloaders(cfg, distributed=False):

    train_dataset, val_dataset = build_dataset(cfg)

    if distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.imgs_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers_per_gpu,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.imgs_per_gpu,
            shuffle=False,
            num_workers=cfg.workers_per_gpu,
            pin_memory=True,
        )
    else:
        val_loader = None

    return train_loader, val_loader
