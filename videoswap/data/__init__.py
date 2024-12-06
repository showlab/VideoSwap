import importlib
from os import path as osp

from videoswap.utils.misc import scandir
from videoswap.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import dataset modules for registry
# scan all the files under the 'datasets' folder and collect files ending with
# '_dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(dataset_folder) if v.endswith('_dataset.py')]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'videoswap.data.{file_name}')
    for file_name in dataset_filenames
]


def build_dataset(dataset_type):
    """Build dataset from options.

    Args:
        opt (dict): Configuration. It must contain:
            dataset_type (str): dataset type.
    """
    dataset = DATASET_REGISTRY.get(dataset_type)
    return dataset
