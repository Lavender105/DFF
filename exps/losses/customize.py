##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Calculate Multi-label Loss (Semantic Loss)"""
import torch
from torch.nn.modules.loss import _Loss

torch_ver = torch.__version__[:3]

__all__ = ['EdgeDetectionReweightedLosses', 'EdgeDetectionReweightedLosses_CPU']


class WeightedCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]
            pad_mask = target[0,:,:]
            target = target[1:,:,:]

            target_nopad = torch.mul(target, pad_mask) # zero out the padding area
            num_pos = torch.sum(target_nopad) # true positive number
            num_total = torch.sum(pad_mask) # true total number
            num_neg = num_total - num_pos
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total) # compute a pos_weight for each image

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = pred - pred * target + log_weight * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())

            loss = loss * pad_mask
            loss = loss.mean()
            loss_total = loss_total + loss

        loss_total = loss_total /  targets.size(0)
        return loss_total

class EdgeDetectionReweightedLosses(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        side5, fuse, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLosses, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss

class EdgeDetectionReweightedLosses_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss
