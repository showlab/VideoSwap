import importlib
from os import path as osp

from videoswap.utils.misc import scandir
from videoswap.utils.registry import PIPELINE_REGISTRY

__all__ = ['build_pipeline']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
pipeline_folder = osp.dirname(osp.abspath(__file__))
pipeline_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(pipeline_folder)
    if v.startswith('trainer_') or v.startswith('pipeline_')
]
# import all the model modules
_pipeline_modules = [
    importlib.import_module(f'videoswap.pipelines.{file_name}')
    for file_name in pipeline_filenames
]


def build_pipeline(pipeline_type):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    pipeline = PIPELINE_REGISTRY.get(pipeline_type)
    return pipeline
