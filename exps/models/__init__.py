from .base import *
from .dff import *
from .casenet import *

def get_edge_model(name, **kwargs):
    models = {
        'dff': get_dff,
        'casenet': get_casenet,
    }
    return models[name.lower()](**kwargs)
