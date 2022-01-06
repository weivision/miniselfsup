# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import time
import apex
import torch
import torch.backends.cudnn as cudnn
from mnssl.utils import AverageMeter, ProgressMeter
from mnssl.utils import is_main_process, save_checkpoint


class BaseTrainer:

    def __init__(self, cfg=None, data_loader=None, 
                 model=None, optimizer=None, 
                 scheduler=None, distributed=False):
        pass

    def train_one_epoch(self):
        pass
    
    def train(self, epoch):
        pass

    def load(self, ckpt_file=None):

        if os.path.isfile(ckpt_file):

            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))
            exit(0)

    def resume(self, ckpt_file=None):

        if os.path.isfile(ckpt_file):

            print("=> resuming checkpoint from '{}'".format(ckpt_file))
            map_location = 'cuda:' + str(torch.distributed.get_rank() % torch.cuda.device_count())
            checkpoint = torch.load(ckpt_file, map_location=map_location)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if self.use_fp16:
                apex.amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))
            exit(0)


class Trainer(BaseTrainer):

    def __init__(self, cfg=None, data_loaders=None, 
                 model=None, optimizer=None, 
                 scheduler=None, distributed=False):
        super(Trainer, self).__init__()

        self.epochs = cfg.epochs
        self.print_freq = cfg.print_freq
        self.save_freq = cfg.save_freq
        self.work_dir = cfg.work_dir

        self.train_loader, self.val_loader = data_loaders
        self.model = model

        self.use_fp16 = cfg.use_fp16
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distributed = distributed
        self.start_epoch = 0

        if self.distributed:
            model_dict = model.module.__dict__
        else:
            model_dict = model.__dict__
        
        if 'update_interval' in model_dict:
            self.update_interval = model_dict['update_interval']
        else:
            self.update_interval = 1
    
    def train_one_epoch(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        iter_per_epoch = self.scheduler.iter_per_epoch
        tail = (iter_per_epoch % self.update_interval)
        tail_start = iter_per_epoch - tail

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, _) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            self.scheduler.iter_step(epoch, i)

            # compute output and loss
            outputs = self.model(images)
            loss = outputs['loss']
            losses.update(loss.item(), images[0].size(0))

            if (i + 1) <= tail_start:
                loss = loss / self.update_interval
            else:
                loss = loss / tail
            
            # compute gradient and do SGD step
            if self.use_fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.distributed:
                self.model.module.optim_update()
            else:
                self.model.optim_update()
            
            if ((i + 1) % self.update_interval == 0) or (i + 1 == iter_per_epoch):
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.distributed:
                self.model.module.iter_update()
            else:
                self.model.iter_update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                progress.display(i)
    
    def train(self):
        
        if self.distributed:
            self.model.module.train_update(self.scheduler)
        else:
            self.model.train_update(self.scheduler)
        
        for epoch in range(self.start_epoch, self.epochs):
            
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            
            self.scheduler.epoch_step(epoch)

            # train for one epoch
            self.train_one_epoch(epoch)

            if self.distributed:
                self.model.module.epoch_update()
            else:
                self.model.epoch_update()

            if (epoch+1) % self.save_freq == 0:
                save_dict = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                }
                if self.use_fp16:
                    save_dict['amp'] = apex.amp.state_dict()
                
                save_checkpoint(save_dict, is_best=False, path=self.work_dir, 
                    filename='checkpoint_{:04d}.pth.tar'.format(epoch))
