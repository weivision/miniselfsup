# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import time

import torch

from mnssl.utils import AverageMeter, ProgressMeter, accuracy, save_checkpoint


class BaseEvaluator:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        distributed=False,
    ):
        pass

    def train_one_epoch(self):
        pass

    def train(self, epoch):
        pass

    def eval(self, epoch):
        pass

    def load(self, ckpt_file=None):

        if os.path.isfile(ckpt_file):
            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            if self.distributed:
                self.model.module.backbone.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.backbone.load_state_dict(checkpoint["state_dict"])
            print("=> loaded pretrained backbone checkpoint '{}'.".format(ckpt_file))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))
            return

    def resume(self, ckpt_file=None):
        if os.path.isfile(ckpt_file):
            print("=> resuming checkpoint from '{}'".format(ckpt_file))

            checkpoint = torch.load(ckpt_file, map_location="cpu")
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_file, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))
            return


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        cfg=None,
        data_loaders=None,
        model=None,
        optimizer=None,
        scheduler=None,
        distributed=False,
    ):
        super(Evaluator, self).__init__()
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.print_freq = cfg.print_freq
        self.eval_freq = cfg.eval_freq
        self.save_freq = cfg.save_freq
        self.work_dir = cfg.work_dir

        self.train_loader, self.val_loader = data_loaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distributed = distributed
        self.start_epoch = 0

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # compute output and loss
            outputs = self.model(images, labels)
            pred = outputs["pred"]
            loss = outputs["loss"]
            losses.update(loss.item(), images[0].size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                progress.display(i)

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):

            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # train for one epoch
            self.train_one_epoch(epoch)

            self.scheduler.epoch_step(epoch)
            
            if (epoch + 1) % self.eval_freq == 0:
                acc1 = self.eval()

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

            if (epoch + 1) % self.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=is_best,
                    path=self.work_dir,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch),
                )

    def eval(self):
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(self.val_loader), [batch_time, losses, top1, top5], prefix="Test: "
        )

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(self.val_loader):

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # compute output
                outputs = self.model(images, labels)
                pred = outputs["pred"]
                loss = outputs["loss"]

                # measure accuracy and record loss
                acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.display(i)

            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

        return top1.avg
