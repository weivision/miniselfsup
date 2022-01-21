# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import errno
import os
import pathlib

from typing import Any, Callable, Optional, Tuple
from torchvision import datasets

from .build import DATASET_REGISTRY


class ImageNetFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        train,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):  
        if not os.path.exists(root):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root)
        
        data_root = os.path.join(root, 'train') if train else os.path.join(root, 'validate')
        super(ImageNetFolder, self).__init__(root=data_root, 
                                       transform=transform,
                                       target_transform=target_transform)
        
        self.multi_crop = isinstance(transform, list)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if self.multi_crop:
                sample = list(map(lambda trans: trans(sample), self.transform))
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


@DATASET_REGISTRY.register()
def imagenet1k(**kwargs):
    return ImageNetFolder(**kwargs)
