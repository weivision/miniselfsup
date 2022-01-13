# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import random

import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps
from torchvision.transforms import InterpolationMode

from .build import TRANSFORM_REGISTRY

normalize = {
    "imagenet1k": transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "imagenet100": transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "cifar10": transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
    "cifar100": transforms.Normalize(mean=[0.507, 0.486, 0.440], std=[0.267, 0.256, 0.276]),
}


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR
    https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"sigma = {self.sigma}, "
        return repr_str


class Solarization(object):
    """Solarization augmentation in BYOL
    https://arxiv.org/abs/2006.07733.
    """

    def __call__(self, x):
        return ImageOps.solarize(x)


@TRANSFORM_REGISTRY.register()
def ssl(dataset, cfg):
    if cfg.type in ["moco", "simsiam"]:
        augmentation = [
            transforms.RandomResizedCrop(cfg.crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize[dataset],
        ]

        transform = transforms.Compose(augmentation)
        ssl_transform = TwoCropsTransform(transform1=transform, transform2=transform)
        return ssl_transform

    elif cfg.type in [
        "byol",
    ]:
        augmentation1 = [
            transforms.RandomResizedCrop(
                cfg.crop_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize[dataset],
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(
                cfg.crop_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            normalize[dataset],
        ]

        transform1 = transforms.Compose(augmentation1)
        transform2 = transforms.Compose(augmentation2)
        ssl_transform = TwoCropsTransform(transform1=transform1, transform2=transform2)
        return ssl_transform

    elif cfg.type in [
        "swav",
    ]:

        color_transform = [
            transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]
            ),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        ]

        trans = []
        for i in range(len(cfg.crop_size)):
            randomresizedcrop = transforms.RandomResizedCrop(
                cfg.crop_size[i],
                scale=(cfg.min_scale_crops[i], cfg.max_scale_crops[i]),
            )
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            normalize[dataset],
                        ]
                    )
                ]
                * cfg.num_crops[i]
            )

        return trans
    else:
        raise Exception("Sorry, can not find transform: {}.".format(cfg.type))


@TRANSFORM_REGISTRY.register()
def train(dataset, cfg):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize[dataset],
        ]
    )


@TRANSFORM_REGISTRY.register()
def val(dataset, cfg):
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.crop_size),
            transforms.ToTensor(),
            normalize[dataset],
        ]
    )
