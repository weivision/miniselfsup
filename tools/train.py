# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
import warnings
from pathlib import Path
import numpy as np

import apex
import torch
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel

import mnssl.utils as utils
from mnssl.utils.config import Config, DictAction
from mnssl.datasets import build_dataloaders
from mnssl.models import build_model
from mnssl.engines import build_optimizer, build_scheduler
from mnssl.engines import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='miniSelfSup training script.')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--port', 
        default=None, 
        type=str,
        help='port used to set up distributed training')
    parser.add_argument(
        '--seed', 
        default=None, 
        type=int,
        help='fix the seed for reproducibility')
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
    train_loader, val_loader = build_dataloaders(cfg.data, 
                                distributed=args.distributed)
    # update dataset statistics
    cfg.scheduler.iter_per_epoch = len(train_loader)
    print(cfg)
    
    # Build model
    model = build_model(cfg.model)

    if args.distributed and cfg.sync_bn:
        # apply sync_bn
        if cfg.sync_bn == "pytorch":
            model = SyncBatchNorm.convert_sync_batchnorm(model)
        elif cfg.sync_bn == "apex":
            # apex syncbn sync bn per group can speeds up 
            # computation compared to global syncbn
            process_group = apex.parallel.create_syncbn_process_group(cfg.syncbn_process_group_size)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        else:
            raise ValueError("Not supported type of BN: {}, must be 'pytorch' or 'apex'!".format(cfg.sync_bn))
    
    model = model.cuda()
    print(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    
    if cfg.use_fp16:
        # init mixed precision
        model, amp_optimizer = apex.amp.initialize(model, optimizer.optimizer, opt_level="O1")
        optimizer.update(amp_optimizer)
        print("Initializing apex mixed precision done.")

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], 
                                        output_device=args.gpu)

    # Build scheduler
    scheduler = build_scheduler(cfg.scheduler, optimizer)
    
    # Build trainer
    trainer = Trainer(cfg, (train_loader, val_loader), model, 
                      optimizer, scheduler, distributed=args.distributed)
    
    if args.resume_from:
        trainer.resume(ckpt_file=args.resume_from)
    
    print("Start training ......")
    # enable cudnn benchmark 
    cudnn.benchmark = True

    trainer.train()
    print("Finished!")


if __name__ == '__main__':
    main()
