from copy import deepcopy

import torch.nn as nn
import torchvision.transforms.functional as F
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import _preprocess_adapter_image
from torchvision.transforms import CenterCrop, Normalize, Resize

from videoswap.utils.registry import TRANSFORM_REGISTRY


def build_transform(opt):
    """Build performance evaluator from options.
    Args:
        opt (dict): Configuration.
    """
    opt = deepcopy(opt)
    transform_type = opt.pop('type')
    transform = TRANSFORM_REGISTRY.get(transform_type)(**opt)
    return transform


TRANSFORM_REGISTRY.register(Normalize)
TRANSFORM_REGISTRY.register(Resize)
TRANSFORM_REGISTRY.register(CenterCrop)


@TRANSFORM_REGISTRY.register()
class ToTensor(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pic):
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@TRANSFORM_REGISTRY.register()
class T2I_Preprocess(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height, self.width = height, width

    def forward(self, pic):
        res = _preprocess_adapter_image(pic, self.height, self.width)
        return res.squeeze(0)
