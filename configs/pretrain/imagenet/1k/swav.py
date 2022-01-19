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
    workers_per_gpu=10,
    train_transform=dict(
        name="ssl",
        type="swav",
        image_size=256,
        crop_size=[224, 96],
        num_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1.0, 0.14],
    ),
    val_transform=None,
)


# model config
model = dict(
    name="SwAV",
    backbone=dict(name="resnet50", zero_init_residual=False, eval_mode=False, padding_mode=True),
    neck=dict(
        name="SwAVNeck",
        normalize=True,
        input_dim=2048,
        hidden_dim=2048,
        output_dim=128,
        nmb_prototypes=3000,
        freeze_prototypes_niters=5005,
    ),
    head=dict(
        name="SwAVHead",
        num_crops=[2, 6],
        crops_for_assign=[0, 1],
        epsilon=0.05,
        temperature=0.1,
        sinkhorn_iterations=3,
        queue_length=3840,
        epoch_queue_starts=15,
        output_dim=128,
    ),
)


# train config
lr = 0.6
epochs = 200
print_freq = 10
save_freq = 10
use_fp16 = True
sync_bn = "pytorch"
save_queue = True


# optimizer config
optimizer = dict(
    name="SwAV",
    type="sgd",
    use_fp16=use_fp16,
    base_lr=lr,
    larc=True,
    momentum=0.9,
    weight_decay=0.000001,
)

# scheduler config
scheduler = dict(
    name="SwAV",
    base_lr=lr,
    final_lr=0.0006,
    start_warmup=0,
    warmup_epochs=0,
    epochs=epochs,
)
