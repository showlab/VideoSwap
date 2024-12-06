import importlib
from os import path as osp

from videoswap.utils.misc import scandir
from videoswap.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'videoswap.models.{file_name}')
    for file_name in model_filenames
]


def build_model(model_type):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    model = MODEL_REGISTRY.get(model_type)
    return model
