##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, BCEWithLogitsLoss
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

torch_ver = torch.__version__[:3]

__all__ = ['GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling', 'SegmentationMultiLosses', 
           'EdgeDetectionReweightedLosses', 'EdgeDetectionReweightedChannelwiseLosses',
           'EdgeDetectionReweightedLosses_CPU', 'EdgeDetectionReweightedChannelwiseLosses_CPU']

class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):

        *preds, target = tuple(inputs)

        loss = super(SegmentationMultiLosses, self).forward(preds[0], target)
        for pred in preds[1:]:
            loss += super(SegmentationMultiLosses, self).forward(pred, target)

        return loss

class WeightedCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        # inputs: tensor (n, 19, 472, 472), n=batch_size_per_gpu
        # targets: tensor (n, 19, 472, 472)
        loss_total = []
        for i in range(targets.size()[0]): # iterate for batch size
            input = inputs[i] #tensor (19, 472, 472)
            target = targets[i] #tensor (19, 472, 472)

            num_pos = torch.sum(target)
            num_neg = target.size()[1] * target.size()[2] - num_pos
            pos_weight = (num_neg / num_pos).clamp(min=1, max=1000) #compute a pos_weight for each image

            max_val = (-input).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log()) #tensor(19, 472, 472)

            if self.reduction == 'none':
                loss_total.append(loss.unsqueeze_(0))
            elif self.reduction == 'elementwise_mean':
                loss_total.append(loss.mean())
            else:
                loss_total.append(loss.sum())

        if self.reduction == 'none':
            loss_total = torch.cat(loss_total, 0)
        elif self.reduction == 'elementwise_mean':
            loss_total = torch.cuda.comm.reduce_add(loss_total) / len(loss_total)
        else:
            loss_total = torch.cuda.comm.reduce_add(loss_total)
        return loss_total

class EdgeDetectionReweightedLosses(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        # import pdb
        # pdb.set_trace()

        side5, fuse, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLosses, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight
        # outputs = [loss_side5]
        # outputs.append(loss_fuse)
        # outputs.append(loss)
        return loss

class EdgeDetectionReweightedLosses_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        import pdb
        pdb.set_trace()

        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight
        # outputs = [loss_side5]
        # outputs.append(loss_fuse)
        # outputs.append(loss)
        return loss

class WeightedChannelwiseCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedChannelwiseCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        # inputs: tensor (n, 19, 472, 472), n=batch_size_per_gpu for gpu version, n=total batch size for cpu version
        # targets: tensor (n, 19, 472, 472)
        loss_total = 0
        for i in range(targets.size()[0]): # iterate for batch size
            input = inputs[i] #tensor (19, 472, 472)
            target = targets[i] #tensor (19, 472, 472)

            loss_c = 0
            for j in range(target.size()[0]): #iterate for number of classes
                input_c = input[j] # tensor (472, 472)
                target_c = target[j] # tensor (472, 472)
                num_pos = torch.sum(target_c)
                if num_pos != 0:
                    num_neg = target_c.size()[0] * target_c.size()[1] - num_pos
                    pos_weight = (num_neg / num_pos).clamp(min=1, max=1000)

                    max_val = (-input_c).clamp(min=0)
                    log_weight = 1 + (pos_weight - 1) * target_c
                    loss = input_c - input_c * target_c + log_weight * (max_val + ((-max_val).exp() + (-input_c - max_val).exp()).log())
                    loss = loss.mean()
                    loss_c += loss
                else:
                    max_val = (-input_c).clamp(min=0)
                    loss = input_c - input_c * target_c + max_val + ((-max_val).exp() + (-input_c - max_val).exp()).log()
                    loss = loss.mean()
                    loss_c +=loss

            loss_total += loss_c / target.size()[0]

        loss_total = loss_total / targets.size()[0]
        return loss_total

class EdgeDetectionReweightedChannelwiseLosses(WeightedChannelwiseCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedChannelwiseLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedChannelwiseLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        side5, fuse, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedChannelwiseLosses, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedChannelwiseLosses, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight
        return loss

class EdgeDetectionReweightedChannelwiseLosses_CPU(WeightedChannelwiseCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedChannelwiseLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedChannelwiseLosses_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        import pdb
        pdb.set_trace()

        # cpu version to debug: 
        # pred:a tuple of 2 tensors, each(n, 19, 472, 472), n is the total batch_size
        # target: a tensor (n, 19, 472, 472)
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedChannelwiseLosses_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedChannelwiseLosses_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight
        return loss

class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)
