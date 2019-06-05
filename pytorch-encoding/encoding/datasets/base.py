###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, crop_size=472, logger=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.crop_size = crop_size
        self.logger = logger

        if self.mode == 'train':
            print('BaseDataset: crop_size {}'.format(crop_size))


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask):
        #hy modified this function
        # center crop
        crop_size = self.crop_size
        w, h = img.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        #np.unpackbits get 24 bits, we extract [:,:5:] and reverse the order (total 19 classes), i.e. [:,:,-1:-20:-1]
        mask = np.unpackbits(np.array(mask), axis=2)[:,:,-1:-20:-1]
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.transpose(0, 1).transpose(0, 2) #channel first
        # final transform
        # return img, self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        #hy modified this function
        # random crop crop_size
        crop_size = self.crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        #np.unpackbits get 24 bits, we extract [:,:5:] and reverse the order (total 19 classes), i.e. [:,:,-1:-20:-1]
        mask = np.unpackbits(np.array(mask), axis=2)[:,:,-1:-20:-1]
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.transpose(0, 1).transpose(0, 2) #channel first
        # return img, self._mask_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(batch[0]))))
