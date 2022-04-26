# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


# dataset config
data = dict(
    dataset="imagenet1k",
    root="/mnt/lustre/share/wli/database/imagenet",
    imgs_per_gpu=64,  # total 64*4=256
    workers_per_gpu=4,
    train_transform=dict(
        name="ssl",
        type="moco",
        image_size=256,
        crop_size=224,
    ),
    val_transform=None,
)


# model config
model = dict(
    name="MoCov3",
    momentum=0.999,
    backbone=dict(
        name="vit_small",
        zero_init_residual=False,
        eval_mode=False,
        padding_mode=False,
    ),
    neck=dict(
        name="MoCov3Neck",
        input_dim=384,
        hidden_dim=4096,
        output_dim=256,
    ),
    head=dict(
        name="MoCov3Head",
        input_dim=256,
        hidden_dim=4096,
        output_dim=256,
    ),
)


# train config
lr = 0.03
epochs = 300
print_freq = 10
save_freq = 10
use_fp16 = False
sync_bn = True
save_queue = False

# optimizer config
optimizer = dict(
    name="MoCo",
    type="sgd",
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001,
)


# scheduler config
scheduler = dict(
    name="MoCo",
    lr=lr,
    epochs=epochs,
)
