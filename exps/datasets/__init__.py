from .base_cityscapes import *
from .cityscapes import CityscapesEdgeDetection
from .sbd import SBDEdgeDetection

datasets = {
    'cityscapes': CityscapesEdgeDetection,
    'sbd': SBDEdgeDetection,
}

def get_edge_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
