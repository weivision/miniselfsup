# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


# dataset config
data = dict(
    dataset="imagenet1k",
    root="/mnt/lustre/share/wli/database/imagenet",
    imgs_per_gpu=128,  # total 128*4=512
    workers_per_gpu=4,
    train_transform=dict(
        name="ssl",
        type="byol",
        image_size=256,
        crop_size=224,
    ),
    val_transform=None,
)


# model config
model = dict(
    name="BYOL",
    update_interval=8,
    base_momentum=0.99,
    end_momentum=1.0,
    backbone=dict(name="resnet50", zero_init_residual=False, eval_mode=False, padding_mode=False),
    neck=dict(
        name="BYOLNeck",
        input_dim=2048,
        hidden_dim=4096,
        output_dim=256,
    ),
    head=dict(
        name="BYOLHead",
        input_dim=256,
        hidden_dim=4096,
        output_dim=256,
    ),
)


# train config
lr = 4.8
epochs = 200
print_freq = 10
save_freq = 10
use_fp16 = False
sync_bn = "pytorch"
save_queue = False


# optimizer config
optimizer = dict(
    name="BYOL",
    type="sgd",
    base_lr=lr,
    larc=True,
    momentum=0.9,
    weight_decay=0.000001,
)

# scheduler config
scheduler = dict(
    name="SwAV",
    base_lr=lr,
    final_lr=0.0,
    start_warmup=0.0,
    warmup_epochs=10,
    epochs=epochs,
)
