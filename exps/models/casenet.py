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


__all__ = ['CaseNet', 'get_casenet']

class CaseNet(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CaseNet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        self.fuse = nn.Conv2d(nclass*4, nclass, 1, groups=nclass, bias=True)

    def forward(self, x):
        c1, c2, c3, _, c5 = self.base_forward(x)

        side1 = self.side1(c1)
        side2 = self.side2(c2)
        side3 = self.side3(c3)
        side5 = self.side5(c5)

        slice5 = side5[:,0:1,:,:]
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:]
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)

        fuse = self.fuse(fuse)

        outputs = [side5, fuse]

        return tuple(outputs)

def get_casenet(dataset='cityscapes', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""CaseNet model from the paper "CASENet: Deep Category-Aware Semantic Edge Detection"
    """
    acronyms = {
        'cityscapes': 'cityscapes',
        'sbd': 'sbd',
    }
    # infer number of classes
    from datasets import datasets
    model = CaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

