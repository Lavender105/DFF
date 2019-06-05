from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CityscapesEdgeDetection
from .cityscapes_orig import CityscapesSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes_orig': CityscapesSegmentation,
    'cityscapes': CityscapesEdgeDetection,
}

def get_segmentation_dataset(name, **kwargs):
    name = name.replace("cityscapes", "cityscapes_orig")
    return datasets[name.lower()](**kwargs)

def get_edge_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
