# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) 2021 MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import mc
import io
import os
import pathlib
import warnings
import errno
from PIL import Image
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from .build import DATASET_REGISTRY


def pil_loader(img_bytes):
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    buff = io.BytesIO(img_bytes)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


class ImageNet(Dataset):

    def __init__(self, root: str, train, transform, read_from='mc', num_classes=1000):
        if not os.path.exists(root):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), root)
        
        root = pathlib.Path(root)
        train_root, train_meta = root / 'train', root / 'meta' / 'train.txt'
        val_root, val_meta = root / 'val', root / 'meta' / 'val.txt'
        
        data_info = (train_root, train_meta) if train else (val_root, val_meta)
        root_dir, meta_file = str(data_info[0]), str(data_info[1])
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.transform = transform
        self.multi_crop = isinstance(transform, list)
        self.read_from = read_from
        
        with open(meta_file) as f:
            lines = f.readlines()
        
        self.num_data = len(lines)
        self.metas = []
        for line in lines:
            img_path, label = line.rstrip().split()
            img_path = os.path.join(self.root_dir, img_path)
            self.metas.append((img_path, int(label)))
        
        self.metas = tuple(self.metas)
        self.targets = tuple(m[1] for m in self.metas)
        
        self.read_from = read_from
        self.initialized = False
    
    def _init_memcached(self):
        if not self.initialized:
            server_list_config = '/mnt/lustre/share/memcached_client/server_list.conf'
            client_config = '/mnt/lustre/share/memcached_client/client.conf'
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config, client_config)
            self.initialized = True
    
    def read_file(self, filepath):
        if self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filepath, value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))
        
        return filebytes
    
    def get_untransformed_image(self, idx):
        return pil_loader(self.read_file(self.metas[idx][0]))
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        img_path, label = self.metas[idx]
        img = pil_loader(self.read_file(img_path))
        if self.transform is not None:
            if self.multi_crop:
                img = list(map(lambda trans: trans(img), self.transform))
            else:
                img = self.transform(img)
        return img, label


@DATASET_REGISTRY.register()
def imagenet1k(**kwargs):
    return ImageNet(num_classes=1000, **kwargs)
