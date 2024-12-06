import random
from typing import List

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from videoswap.utils.registry import MODEL_REGISTRY


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.SiLU(inplace=False),
            nn.Linear(mid_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.mlp(x)


def bilinear_interpolation(level_adapter_state, x, y, frame_idx, interpolated_value):
    # note the boundary
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1
    x_frac = x - x1
    y_frac = y - y1

    x1, x2 = max(min(x1, level_adapter_state.shape[3] - 1), 0), max(min(x2, level_adapter_state.shape[3] - 1), 0)
    y1, y2 = max(min(y1, level_adapter_state.shape[2] - 1), 0), max(min(y2, level_adapter_state.shape[2] - 1), 0)

    w11 = (1 - x_frac) * (1 - y_frac)
    w21 = x_frac * (1 - y_frac)
    w12 = (1 - x_frac) * y_frac
    w22 = x_frac * y_frac

    level_adapter_state[frame_idx, :, y1, x1] += interpolated_value * w11
    level_adapter_state[frame_idx, :, y1, x2] += interpolated_value * w21
    level_adapter_state[frame_idx, :, y2, x1] += interpolated_value * w12
    level_adapter_state[frame_idx, :, y2, x2] += interpolated_value * w22

    return level_adapter_state


@MODEL_REGISTRY.register()
class SparsePointAdapter(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        embedding_channels=1280,
        channels=[320, 640, 1280, 1280],
        downsample_rate=[8, 16, 32, 64],
        mid_dim=128
    ):
        super().__init__()

        self.model_list = nn.ModuleList()

        for ch in channels:
            self.model_list.append(MLP(embedding_channels, ch, mid_dim))

        self.downsample_rate = downsample_rate
        self.channels = channels
        self.radius = 2

    def generate_loss_mask(self, point_index_list, point_tracker, num_frames, h, w, loss_type):
        if loss_type == 'global':
            loss_mask = torch.ones((num_frames, 4, h // self.downsample_rate[0], w // self.downsample_rate[0]))
        else:
            loss_mask = torch.zeros((num_frames, 4, h // self.downsample_rate[0], w // self.downsample_rate[0]))
            for point_idx in point_index_list:
                for frame_idx in range(num_frames):
                    px, py = point_tracker[frame_idx, point_idx]

                    if px < 0 or py < 0:
                        continue
                    else:
                        px, py = px / self.downsample_rate[0], py / self.downsample_rate[0]

                        x1 = int(px) - self.radius
                        y1 = int(py) - self.radius
                        x2 = int(px) + self.radius
                        y2 = int(py) + self.radius

                        x1, x2 = max(min(x1, loss_mask.shape[3] - 1), 0), max(min(x2, loss_mask.shape[3] - 1), 0)
                        y1, y2 = max(min(y1, loss_mask.shape[2] - 1), 0), max(min(y2, loss_mask.shape[2] - 1), 0)

                        loss_mask[:, :, y1:y2, x1:x2] = 1.0
        return loss_mask

    def forward(self, point_tracker, size, point_embedding, index_list=None, drop_rate=0.0, loss_type='global') -> List[torch.Tensor]:

        point_tracker = point_tracker.squeeze(0)
        point_embedding = point_embedding.squeeze(0)

        w, h = size
        num_frames, num_points = point_tracker.shape[:2]

        if self.training:
            point_index_list = [point_idx for point_idx in range(num_points) if random.random() > drop_rate]
            loss_mask = self.generate_loss_mask(point_index_list, point_tracker, num_frames, h, w, loss_type)
        else:
            point_index_list = [point_idx for point_idx in range(num_points) if index_list is None or point_idx in index_list]

        adapter_state = []
        for level_idx, module in enumerate(self.model_list):

            downsample_rate = self.downsample_rate[level_idx]
            level_w, level_h = w // downsample_rate, h // downsample_rate

            point_feat = module(point_embedding)

            level_adapter_state = torch.zeros((num_frames, self.channels[level_idx], level_h, level_w)).to(point_feat.device, dtype=point_feat.dtype)

            for point_idx in point_index_list:

                for frame_idx in range(num_frames):
                    px, py = point_tracker[frame_idx, point_idx]

                    if px < 0 or py < 0:
                        continue
                    else:
                        px, py = px / downsample_rate, py / downsample_rate
                        level_adapter_state = bilinear_interpolation(level_adapter_state, px, py, frame_idx, point_feat[point_idx])
            adapter_state.append(level_adapter_state)

        if self.training:
            return adapter_state, loss_mask
        else:
            return adapter_state
