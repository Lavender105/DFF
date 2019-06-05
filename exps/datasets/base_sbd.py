###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=472, 
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented


    def _testval_sync_transform(self, img):
        crop_size = self.crop_size
        w, h = img.size
        img = np.array(img).astype(np.float)

        short_size = min(w, h)
        if short_size < crop_size:
            padh = crop_size - h if h < crop_size else 0
            padw = crop_size - w if w < crop_size else 0
            
            img_r = img[:,:,0]
            img_r = np.pad(img_r, ((0, padh),(0, padw)), 'constant', constant_values=((.485*255, .485*255), (.485*255, .485*255)))
            img_g = img[:,:,1]
            img_g = np.pad(img_g, ((0, padh),(0, padw)), 'constant', constant_values=((.456*255, .456*255), (.456*255, .456*255)))
            img_b = img[:,:,2]
            img_b = np.pad(img_b, ((0, padh),(0, padw)), 'constant', constant_values=((.406*255, .406*255), (.406*255, .406*255)))
            img = np.array([img_r, img_g, img_b])
            img = np.transpose(img, (1, 2, 0))/255
        return img


    def _vis_sync_transform(self, img, mask):
        crop_size = self.crop_size
        w, h = img.size
        img = np.array(img).astype(np.float)

        short_size = min(w, h)
        if short_size < crop_size:
            padh = crop_size - h if h < crop_size else 0
            padw = crop_size - w if w < crop_size else 0
            
            img_r = img[:,:,0]
            img_r = np.pad(img_r, ((0, padh),(0, padw)), 'constant', constant_values=((.485*255, .485*255), (.485*255, .485*255)))
            img_g = img[:,:,1]
            img_g = np.pad(img_g, ((0, padh),(0, padw)), 'constant', constant_values=((.456*255, .456*255), (.456*255, .456*255)))
            img_b = img[:,:,2]
            img_b = np.pad(img_b, ((0, padh),(0, padw)), 'constant', constant_values=((.406*255, .406*255), (.406*255, .406*255)))
            img = np.array([img_r, img_g, img_b])
            img = np.transpose(img, (1, 2, 0))/255

        mask = np.unpackbits(np.array(mask), axis=2)[:,:,-1:-21:-1]
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.permute(2, 0, 1)

        return img, mask


    def _val_sync_transform(self, img, mask):
        # center crop
        crop_size = self.crop_size
        w, h = img.size
        img = np.array(img).astype(np.float)
        pad_index = np.ones([h, w])

        short_size = min(w, h)
        if short_size < crop_size:
            padh = crop_size - h if h < crop_size else 0
            padw = crop_size - w if w < crop_size else 0
            
            img_r = img[:,:,0]
            img_r = np.pad(img_r, ((0, padh),(0, padw)), 'constant', constant_values=((.485*255, .485*255), (.485*255, .485*255)))
            img_g = img[:,:,1]
            img_g = np.pad(img_g, ((0, padh),(0, padw)), 'constant', constant_values=((.456*255, .456*255), (.456*255, .456*255)))
            img_b = img[:,:,2]
            img_b = np.pad(img_b, ((0, padh),(0, padw)), 'constant', constant_values=((.406*255, .406*255), (.406*255, .406*255)))
            img = np.array([img_r, img_g, img_b])
            img = np.transpose(img, (1, 2, 0))

            pad_index = np.pad(pad_index, ((0, padh),(0, padw)), 'constant', constant_values=((0, 0), (0, 0)))

            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=(255, 255, 255))

        w, h = mask.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        img = img[y1:y1+crop_size, x1:x1+crop_size, :]/255
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        pad_index = pad_index[y1:y1+crop_size, x1:x1+crop_size]

        pad_index = pad_index.reshape(-1, pad_index.shape[0], pad_index.shape[1])
        pad_index = torch.from_numpy(pad_index).float()

        mask = np.unpackbits(np.array(mask), axis=2)[:,:,-1:-21:-1]
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.permute(2, 0, 1) #channel first
        
        mask = torch.cat((pad_index, mask), dim=0) #(21, crop_size, crop_size)

        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size

        w, h = img.size
        short_size = min(w, h)

        img = np.array(img).astype(np.float)
        pad_index = np.ones([h, w])

        # pad crop
        if short_size < crop_size:
            padh = crop_size - h if h < crop_size else 0
            padw = crop_size - w if w < crop_size else 0
            # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            img_r = img[:,:,0]
            img_r = np.pad(img_r, ((0, padh),(0, padw)), 'constant', constant_values=((.485*255, .485*255), (.485*255, .485*255)))
            img_g = img[:,:,1]
            img_g = np.pad(img_g, ((0, padh),(0, padw)), 'constant', constant_values=((.456*255, .456*255), (.456*255, .456*255)))
            img_b = img[:,:,2]
            img_b = np.pad(img_b, ((0, padh),(0, padw)), 'constant', constant_values=((.406*255, .406*255), (.406*255, .406*255)))
            img = np.array([img_r, img_g, img_b])
            img = np.transpose(img, (1, 2, 0))

            pad_index = np.pad(pad_index, ((0, padh),(0, padw)), 'constant', constant_values=((0, 0), (0, 0)))

            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=(255, 255, 255))
        
        # random crop crop_size
        w, h = mask.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img[y1:y1+crop_size, x1:x1+crop_size, :]/255 # crop image
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size)) # crop mask
        pad_index = pad_index[y1:y1+crop_size, x1:x1+crop_size] # crop pad_index
        
        pad_index = pad_index.reshape(-1, pad_index.shape[0], pad_index.shape[1])
        pad_index = torch.from_numpy(pad_index).float()

        mask = np.unpackbits(np.array(mask), axis=2)[:,:,-1:-21:-1] #(crop_size, crop_size, 20)
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.permute(2, 0, 1) #channel first
        
        mask = torch.cat((pad_index, mask), dim=0) #(21, crop_size, crop_size)

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
