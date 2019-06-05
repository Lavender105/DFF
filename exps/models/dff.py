###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from . import BaseNet


__all__ = ['DFF', 'get_dff']

class DFF(BaseNet):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion 
        for Semantic Edge Detection" *IJCAI*, 2019

    """
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass

        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*4, nclass*4, norm_layer=norm_layer)

        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))

        self.side5_w = nn.Sequential(nn.Conv2d(2048, nclass*4, 1, bias=True),
                                   norm_layer(nclass*4),
                                   nn.ConvTranspose2d(nclass*4, nclass*4, 16, stride=8, padding=4, bias=False))

    def forward(self, x):
        c1, c2, c3, _, c5 = self.base_forward(x)
        
        side1 = self.side1(c1) # (N, 1, H, W)
        side2 = self.side2(c2) # (N, 1, H, W)
        side3 = self.side3(c3) # (N, 1, H, W)
        side5 = self.side5(c5) # (N, 19, H, W)
        side5_w = self.side5_w(c5) # (N, 19*4, H, W)
        
        ada_weights = self.ada_learner(side5_w) # (N, 19, 4, H, W)

        slice5 = side5[:,0:1,:,:] # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:] # (N, 1, H, W)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1) # (N, 19*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, 19, 4, H, W)
        fuse = torch.mul(fuse, ada_weights) # (N, 19, 4, H, W)
        fuse = torch.sum(fuse, 2) # (N, 19, H, W)

        outputs = [side5, fuse]

        return tuple(outputs)


class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x) # (N, 19*4, H, W)
        x = self.conv2(x) # (N, 19*4, H, W)
        x = self.conv3(x) # (N, 19*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3)) # (N, 19, 4, H, W)
        return x


def get_dff(dataset='cityscapes', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""DFF model from the paper "Dynamic Feature Fusion for Semantic Edge Detection"
    """
    acronyms = {
        'cityscapes': 'cityscapes',
        'sbd': 'sbd',
    }
    # infer number of classes
    from datasets import datasets
    model = DFF(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

