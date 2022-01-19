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
        type="simsiam",
        image_size=256,
        crop_size=224,
    ),
    val_transform=None,
)


# model config
model = dict(
    name="SimSiam",
    backbone=dict(name="resnet50", zero_init_residual=True, eval_mode=False, padding_mode=False),
    neck=dict(
        name="SimSiamNeck",
        input_dim=2048,
        output_dim=2048,
    ),
    head=dict(
        name="SimSiamHead",
        input_dim=2048,
        hidden_dim=512,
        output_dim=2048,
    ),
)


# train config
lr = 0.05
epochs = 100
print_freq = 10
save_freq = 10

use_fp16 = False
sync_bn = "pytorch"
save_queue = False

# optimizer config
optimizer = dict(
    name="SimSiam",
    type="sgd",
    lr=lr,
    fix_head_lr=True,
    momentum=0.9,
    weight_decay=0.0001,
)

# scheduler config
scheduler = dict(
    name="SimSiam",
    lr=lr,
    epochs=epochs,
)
