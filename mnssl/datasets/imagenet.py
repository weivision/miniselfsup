# ------------------------------------------------------------------------
# miniSelfSup
# Copyright (c) MMLab@NTU. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import errno
import io
import os
import pathlib
import warnings
import zipfile

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZipReader(object):
    """A class to read zipped files"""

    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, "r")
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def read(zip_path, data_path):
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(data_path)
        return data


def img_loader(img_bytes):
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    buff = io.BytesIO(img_bytes)
    with Image.open(buff) as img:
        img = img.convert("RGB")
    return img


class ImageNet(Dataset):
    def __init__(self, root: str, train, transform=None, read_from="zip", num_classes=1000):

        if not os.path.exists(root):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root)

        root = pathlib.Path(root)
        train_zip, train_meta = root / "train.zip", root / "meta" / "train.txt"
        val_zip, val_meta = root / "val.zip", root / "meta" / "val.txt"

        data_info = (train_zip, train_meta) if train else (val_zip, val_meta)
        root_file, meta_file = str(data_info[0]), str(data_info[1])

        self.num_classes = num_classes
        self.root_file = root_file
        self.transform = transform
        self.multi_crop = isinstance(transform, list)
        self.read_from = read_from

        with open(meta_file) as f:
            lines = f.readlines()

        self.num_data = len(lines)
        self.metas = []
        for line in lines:
            img_path, label = line.rstrip().split()
            self.metas.append((img_path, int(label)))

        self.metas = tuple(self.metas)
        self.targets = tuple(m[1] for m in self.metas)

        self.read_from = read_from

    def read_file(self, img_path):

        imgbytes = ZipReader.read(self.root_file, img_path)

        return imgbytes

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img_path, label = self.metas[idx]
        img = img_loader(self.read_file(img_path))
        if self.transform is not None:
            if self.multi_crop:
                img = list(map(lambda trans: trans(img), self.transform))
            else:
                img = self.transform(img)
        return img, label


@DATASET_REGISTRY.register()
def imagenet1k(**kwargs):
    return ImageNet(num_classes=1000, **kwargs)
