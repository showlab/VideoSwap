from videoswap.models.animatediff_models.unet import AnimateDiffUNet3DModel
from videoswap.utils.registry import MODEL_REGISTRY

MODEL_REGISTRY.register(AnimateDiffUNet3DModel)
