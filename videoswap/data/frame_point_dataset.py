from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from videoswap.data.transform import build_transform
from videoswap.utils.registry import DATASET_REGISTRY


def select_frame_idx(begin_frame_idx, end_frame_idx, n):
    total_frames = end_frame_idx - begin_frame_idx
    frame_interval = total_frames // (n - 1)
    selected_frames = []

    for i in range(n):
        frame_idx = int(begin_frame_idx + i * frame_interval)
        selected_frames.append(frame_idx)

    return selected_frames


@DATASET_REGISTRY.register()
class SingleVideoPointDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt

        self.total_frames = sorted(list(Path(opt['path']).iterdir()))

        self.select_id = select_frame_idx(0, min(len(self.total_frames), opt['total_frames']), opt['num_frames'])

        # video frames
        self.video = [self.total_frames[i] for i in self.select_id]
        self.prompt = opt['prompt']
        self.num_video = opt.get('dataset_enlarge_ratio', 1)

        self.video_transform = \
            transforms.Compose([build_transform(transform_opt) for transform_opt in opt['video_transform']])

        frames = [Image.open(path).convert('RGB') for path in self.video]
        frames = torch.stack([self.video_transform(frame) for frame in frames])
        self.frames = rearrange(frames, 'f c h w -> c f h w')
        self.size_y, self.size_x = self.frames.shape[-2:]

        if 'tap_path' in opt:
            self.condition = self.get_conditions(opt['tap_path'])
        else:
            self.condition = None

    def __len__(self):
        return self.num_video

    def get_frames(self):
        video_transform = \
            transforms.Compose([build_transform(transform_opt) for transform_opt in self.opt['video_transform'] if transform_opt['type'] not in ('ToTensor', 'Normalize')])
        frames = [video_transform(Image.open(path).convert('RGB')) for path in self.video]
        return frames

    def get_conditions(self, tap_path=None):
        if tap_path is None:
            return self.condition
        else:
            TAP = torch.load(tap_path)
            pred_tracks, point_name2id, point_embedding = TAP['pred_tracks'], TAP['point_name2id'], TAP['point_embedding']
            select_pred_tracks = pred_tracks[self.select_id]
            assert pred_tracks.shape[1] == point_embedding.shape[0]
            return {'pred_tracks': select_pred_tracks, 'point_embedding': point_embedding, 'point_name2id': point_name2id, 'img_size': (self.size_x, self.size_y)}

    def __getitem__(self, index):

        return_batch = {
            'images': self.frames,
            'prompt': self.prompt,
        }
        if self.condition is not None:
            return_batch.update(self.condition)
        return return_batch
