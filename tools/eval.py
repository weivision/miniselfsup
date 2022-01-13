# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import argparse
from pathlib import Path

import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel

import mnssl.utils as utils
from mnssl.datasets import build_dataloaders
from mnssl.engines import Evaluator, build_optimizer, build_scheduler
from mnssl.models import build_model
from mnssl.utils.config import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description="miniSelfSup linear evaluation script.")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--load_from", help="the checkpoint file to load from")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--port", default=None, type=str, help="port used to set up distributed training"
    )
    parser.add_argument("--seed", default=None, type=int, help="fix the seed for reproducibility")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.config:
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
    else:
        print("Not using configuration file!")
        exit(0)

    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        cfg.work_dir = args.work_dir

    utils.init_distributed_mode(args)
    utils.init_rand_seed(args.seed)
    print(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # Build dataset and loaders
    train_loader, val_loader = build_dataloaders(cfg.data, distributed=args.distributed)

    # Build model
    model = build_model(cfg.model)

    if args.distributed:
        # apply sync_bn
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    print(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # Build scheduler
    scheduler = build_scheduler(cfg.scheduler, optimizer)

    # Build trainer
    evaluator = Evaluator(
        cfg, (train_loader, val_loader), model, optimizer, scheduler, distributed=args.distributed
    )

    if args.load_from:
        evaluator.load(ckpt_file=args.load_from)

    if args.resume_from:
        evaluator.resume(ckpt_file=args.resume_from)

    print("Start training ......")
    # enable cudnn benchmark
    cudnn.benchmark = True
    evaluator.train()
    print("Finished!")


if __name__ == "__main__":
    main()
