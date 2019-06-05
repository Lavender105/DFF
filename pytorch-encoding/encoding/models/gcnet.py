###########################################################################
# Created by: CASIA IVA 
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
from collections import OrderedDict
import encoding

__all__ = ['GCNet', 'get_gcnet']

class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.conv2(h)
        return h

class Customized_Unit(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1, norm_type=0):
        super(Customized_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        # self.gcn_bn = nn.BatchNorm1d(1, momentum=0.1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: expend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1),
                              groups=1, bias=False)

        self.norm_type = norm_type
        if self.norm_type % 2 == 0:
            self.gamma = Parameter(torch.zeros(1))
        elif self.norm_type % 2 == 1:
            # self.blocker = encoding.nn.BatchNorm1d(1)
            self.blocker = nn.BatchNorm1d(1, eps=1e-04, momentum=0.1)
        else:
            assert 0

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        # x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=2)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.norm_type // 2 == 0:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2)) # TODO: Grid search
        elif self.norm_type // 2 == 1:
            x_n_state = x_n_state * (1000. / x_state_reshaped.size(2))
        else:
            assert 0
        # x_n_state = self.gcn_bn(x_n_state.view(batch_size, 1, -1)).view(batch_size, self.num_s, self.num_n)

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        # x_n_rel = self.GCN_G(x_n_state.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        # x_n_rel = x_n_rel.contiguous() + x_n_state
        # # (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        if hasattr(self, 'gamma'):
           out = x + self.gamma * self.fc_2(x_state)
        elif hasattr(self, 'blocker'):
           out = x + self.blocker(self.fc_2(x_state).view(batch_size, 1, -1)).view(x.shape)
        else:
            assert 0

        return out

class GCNet(BaseNet):
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
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, gcn_search=None, **kwargs):
        super(GCNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = GCNetHead(2048, nclass, norm_layer, gcn_search)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)

        outputs = []
        for i in range(len(x)):
            outputs.append(upsample(x[i], imsize, **self._up_kwargs))

        return tuple(outputs)
        
class GCNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, gcn_search):
        super(GCNetHead, self).__init__()

        inter_channels = in_channels // 4
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        if gcn_search['num_blocks'] > 0: 
            self.gcn = nn.Sequential(OrderedDict([ ("GCN%02d"%i, 
                          Customized_Unit(inter_channels, gcn_search['inner_plane'], norm_type=gcn_search['norm_type'],  kernel=gcn_search['kernel_size'])
                          ) for i in range(gcn_search['num_blocks']) ]))

        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat = self.conv51(x)
        if hasattr(self, 'gcn'):
           gc_feat = self.gcn(feat)
        else:
            gc_feat = feat
        gc_conv = self.conv52(gc_feat)
        gc_output = self.conv6(gc_conv)

        output = [gc_output]
        return tuple(output)


def get_gcnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""GCNet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = GCNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

