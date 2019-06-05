###########################################################################
# Created by: CASIA IV.
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import upsample,normalize
from ..models import BaseNet


__all__ = ['CaseNet', 'get_casenet']

class CaseNet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

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

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CaseNet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        self.slice_fuse = nn.Conv2d(4, 1, 1, bias=True)

    def forward(self, x):
        # import pdb #hy added
        # pdb.set_trace() #hy added

        c1, c2, c3, _, c5 = self.base_forward(x)

        side1 = self.side1(c1)
        side2 = self.side2(c2)
        side3 = self.side3(c3)
        side5 = self.side5(c5)

        fuse_list = []
        for i in range(self.nclass):
            index = torch.tensor([i]).cuda()
            slice5 = torch.index_select(side5, 1, index) #the first dimension is batch_size, the second dimension is channel
            fuse = torch.cat((slice5, side1, side2, side3), 1)
            fuse = self.slice_fuse(fuse)
            fuse_list.append(fuse)
        fuse_final = torch.cat(fuse_list, 1)

        outputs = [side5]
        outputs.append(fuse_final)

        return tuple(outputs)
        
# class CaseNetHead(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer):
#         super(CaseNetHead, self).__init__()
#         self.side1 = nn.Conv2d(in_channels[0], 1, 1)
#         self.side2 = nn.Sequential(nn.Conv2d(in_channels[1], 1, 1),
#                                    nn.ConvTranspose2d(1, 1, 4, stride=2))
#         self.side3 = nn.Sequential(nn.Conv2d(in_channels[2], 1, 1),
#                                    nn.ConvTranspose2d(1, 1, 8, stride=4))
#         self.side5 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, 1),
#                                    nn.ConvTranspose2d(out_channels, out_channels, 16, stride=8))
#         fuse_list = []
#         for i in range(out_channels):
#             index = torch.tensor([i])
#             slice5 = torch.index_select(side5, 1, index) #the first dimension is batch_size, the second dimension is channel?
#             fuse = torch.cat((slice5, self.side1, self.side2, self.side5), 1)
#             fuse_list.append(fuse)
#         self.fuse = torch.cat(fuse_list, 1)


#         inter_channels = in_channels // 4
#         self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                    norm_layer(inter_channels),
#                                    nn.ReLU())
        
#         self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                    norm_layer(inter_channels),
#                                    nn.ReLU())

#         self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

#     def forward(self, x):
#         side1 = self.side1(x)
#         side2 = self.side2(x)

#         feat = self.conv51(x)
#         gc_feat = feat
#         gc_conv = self.conv52(gc_feat)
#         gc_output = self.conv6(gc_conv)

#         output = [gc_output]
#         output.append(gc_output)
#         output.append(gc_output)
#         return tuple(output)


def get_casenet(dataset='cityscapes', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""CaseNet model from the paper "CASENet: Deep Category-Aware Semantic Edge Detection"
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ..datasets import datasets
    model = CaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

