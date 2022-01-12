# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


# dataset config
data = dict(
    dataset="imagenet1k",
    root="/mnt/lustre/share/wli/database/imagenet",
    imgs_per_gpu=512,  # total 512*8=4096
    workers_per_gpu=5,
    train_only=True,
    train_transform=dict(
        name="train",
        image_size=256,
        crop_size=224,
    ),
    val_transform=dict(
        name="val",
        image_size=256,
        crop_size=224,
    ),
)


# model config
model = dict(
    name="LinearCls",
    backbone=dict(
        name="resnet50",
        zero_init_residual=False,
        eval_mode=False,
        padding_mode=False,
    ),
    neck=None,
    head=dict(
        name="ClsHead",
        input_dim=2048,
        num_classes=1000,
    ),
)


# train config
lr = 1.6
epochs = 90
print_freq = 10
eval_freq = 1
save_freq = 10


# optimizer config
optimizer = dict(
    name="LinearCls",
    larc=True,
    type="sgd",
    lr=lr,
    momentum=0.9,
    weight_decay=0.0,
    epochs=epochs,
)


# scheduler config
scheduler = dict(
    name="LinearCls",
    lr=lr,
    epochs=epochs,
)
