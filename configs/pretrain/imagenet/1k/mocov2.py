# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


# dataset config
data = dict(
    dataset='imagenet1k',
    root='/mnt/lustre/share/wli/database/imagenet',
    imgs_per_gpu=64,  # total 64*4=256
    workers_per_gpu=4,
    train_transform=dict(
        name='ssl',
        type='moco',
        image_size=256,
        crop_size=224,
    ),
    val_transform=None,
)


# model config
model = dict(
    name='MoCo',
    momentum=0.999,
    backbone=dict(
        name='resnet50',
        zero_init_residual=False,
        eval_mode=False,
        padding_mode=False,),
    neck=dict(
        name='MoCoNeck',
        input_dim=2048,
        hid_dim=2048,
        output_dim=128,
    ),
    head=dict(
        name='MoCoHead',
        feat_dim=128,
        temperature=0.2,
        queue_len=65536,
    ),
)


# train config
lr=0.03
epochs=200
print_freq=10
save_freq=10
use_fp16=False
sync_bn=None

# optimizer config
optimizer = dict(
    name='MoCo',
    type='sgd',
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001,
)


# scheduler config
scheduler = dict(
    name='MoCo',
    lr=lr,
    epochs=epochs,
)
