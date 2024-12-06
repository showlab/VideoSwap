import argparse
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, './thirdparty/unimatch')
from unimatch.unimatch import UniMatch  # noqa


class UniMatchWrapper():
    def __init__(self, model_path):
        args = argparse.Namespace()
        args.feature_channels = 128
        args.upsample_factor = 4
        args.num_scales = 2
        args.num_head = 1
        args.num_reg_refine = 6
        args.ffn_dim_expansion = 4
        args.num_transformer_layers = 6
        args.reg_refine = True
        args.task = 'flow'

        self.model = UniMatch(
            feature_channels=args.feature_channels,
            num_scales=args.num_scales,
            upsample_factor=args.upsample_factor,
            num_head=args.num_head,
            ffn_dim_expansion=args.ffn_dim_expansion,
            num_transformer_layers=args.num_transformer_layers,
            reg_refine=args.reg_refine,
            task=args.task
        ).to(device)
        self.model.load_state_dict(torch.load(model_path)['model'])
        self.model.to(device)
        self.model.eval()

        self.attn_type = 'swin'
        self.num_reg_refine = 6
        self.corr_radius_list = [-1, 4]
        self.attn_splits_list = [2, 8]
        self.prop_radius_list = [-1, 1]
        self.pred_bidir_flow = True
        self.pred_bwd_flow = False
        self.padding_factor = 32
        self.max_long_edge = 768

    def compute_flow(self, fn1, fn2):
        """ load and resize to multiple of 64 """
        image1 = Image.open(fn1)
        image2 = Image.open(fn2)

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        im_h = image1.shape[0]
        im_w = image1.shape[1]
        long_edge = max(im_w, im_h)
        factor = long_edge / self.max_long_edge
        if factor > 1:
            new_w = int(im_w // factor)
            new_h = int(im_h // factor)
            image1 = cv2.resize(image1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True
        else:
            transpose_img = False

        nearest_size = [int(np.ceil(image1.size(-2) / self.padding_factor)) * self.padding_factor,
                        int(np.ceil(image1.size(-1) / self.padding_factor)) * self.padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size
        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

        pred_bwd_flow = False
        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = self.model(
            image1, image2,
            attn_type=self.attn_type,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            num_reg_refine=self.num_reg_refine,
            task='flow',
            pred_bidir_flow=True
        )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)

            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow_pr = rearrange(flow_pr, 'b c h w -> b h w c').detach().cpu().numpy()

        return flow_pr


def preprocess(args):
    files = sorted(args.video_path.glob('*.jpg'))
    args.flow_save_path.mkdir(exist_ok=True)

    unimatch_wrapper = UniMatchWrapper(
        model_path='thirdparty/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')

    for i, file1 in enumerate(tqdm(files, desc='computing flow')):
        if i < len(files) - 1:
            file2 = files[i + 1]
            flow_bidir = unimatch_wrapper.compute_flow(file1, file2)
            filename = osp.splitext(osp.basename(file1))[0]
            save_path = osp.join(args.flow_save_path, f'{filename}.npy')
            np.save(save_path, flow_bidir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument('--video_path', type=Path, default=Path('datasets/paper_evaluation/object/1070375476/frames'), help='folder to process')
    parser.add_argument('--flow_save_path', type=Path, default=Path('datasets/paper_evaluation/object/1070375476/flows'), help='folder to process')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    preprocess(args=args)
