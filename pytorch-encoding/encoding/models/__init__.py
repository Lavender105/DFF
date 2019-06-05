from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
# from .danet import *
# from .plain import *
from .casenet import *

def get_edge_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        # 'danet': get_danet,
        # 'plain': get_plain,
        'casenet': get_casenet,
    }
    return models[name.lower()](**kwargs)
