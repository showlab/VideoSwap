import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.data
from einops import repeat
from omegaconf import OmegaConf
from tqdm import tqdm

from videoswap.atlas.implicit_neural_networks import IMLP_Hash
from videoswap.atlas.unwrap_utils import load_input_data
from videoswap.utils.vis_util import visualize_point_sequence


def compute_Wm(xyt, func, deltax, deltay):
    xplus1_y_t, x_yplus1_t = xyt.clone(), xyt.clone()
    xplus1_y_t[:, 0] = xyt[:, 0] + deltax
    x_yplus1_t[:, 1] = xyt[:, 1] + deltay

    uv = func(xyt)
    uv_xplus1_y = func(xplus1_y_t)
    uv_x_yplus1 = func(x_yplus1_t)

    unit_transf_by_dx = (uv_xplus1_y - uv) / deltax
    unit_transf_by_dy = (uv_x_yplus1 - uv) / deltay

    unit_transf_by_dx_dy = torch.cat([unit_transf_by_dx.unsqueeze(1), unit_transf_by_dy.unsqueeze(1)], dim=1)
    return unit_transf_by_dx_dy


def propogate_point(x, y, target_x, target_y, t, number_of_frames, FG_UV_Mapping, FG_UV_Mapping_Inverse, norm_Scoord_func, norm_Tcoord_func, device):
    '''
    given the x y and target_x and target_y at timestep t, propogate this relation to all T
    '''

    xyt_foreground = torch.tensor([norm_Scoord_func(x), norm_Scoord_func(y), norm_Tcoord_func(t)]).float().unsqueeze(0).to(device)
    uv_foreground = FG_UV_Mapping(xyt_foreground)

    unit_transf_by_dx_dy = compute_Wm(xyt_foreground, func=FG_UV_Mapping, deltax=0.1, deltay=0.05)

    dx_dy = torch.tensor([[norm_Scoord_func(target_x) - norm_Scoord_func(x), norm_Scoord_func(target_y) - norm_Scoord_func(y)]]).float().unsqueeze(0).to(device)
    delta_uv = dx_dy.bmm(unit_transf_by_dx_dy)

    T = torch.arange(number_of_frames).unsqueeze(-1).to(device)
    uv_foreground_allT = repeat(uv_foreground, 'b c -> (b t) c', t=T.shape[0])
    uvt_foreground_allT = torch.cat([uv_foreground_allT, norm_Tcoord_func(T)], dim=-1)

    unit_transf_by_du_dv = compute_Wm(uvt_foreground_allT, func=FG_UV_Mapping_Inverse, deltax=0.1, deltay=0.05)[..., :2]
    delta_uv_allT = repeat(delta_uv, 'b c d -> (b t) c d', t=T.shape[0])
    delta_xy_target = delta_uv_allT.bmm(unit_transf_by_du_dv)

    return delta_xy_target.squeeze(1)


def init_atlas_model(atlas_config, atlas_model):

    checkpoint = torch.load(atlas_model)

    # build model
    FG_UV_Mapping = IMLP_Hash(**atlas_config['models']['FG_UV_Mapping']).to(device)
    FG_UV_Mapping.load_state_dict(checkpoint['FG_UV_Mapping'])

    F_Alpha = IMLP_Hash(**atlas_config['models']['F_Alpha']).to(device)
    F_Alpha.load_state_dict(checkpoint['F_Alpha'])

    FG_UV_Mapping_Inverse = IMLP_Hash(**atlas_config['models']['FG_UV_Mapping_Inverse']).to(device)
    FG_UV_Mapping_Inverse.load_state_dict(checkpoint['FG_UV_Mapping_Inverse'])

    return FG_UV_Mapping, FG_UV_Mapping_Inverse, F_Alpha


def propagate_point_sequence(
    source_point_path, source_tap_path, target_point_path,
    FG_UV_Mapping, FG_UV_Mapping_Inverse, F_Alpha,
    larger_dim, number_of_frames, norm_Scoord_func, norm_Tcoord_func
):

    with open(source_point_path, 'r') as fr:
        source_point_dict = json.load(fr)
        keyframe_timestep = int(osp.splitext(osp.basename(source_point_path))[0])

    source_tap = torch.load(source_tap_path)

    pred_tracks, point_name2id = source_tap['pred_tracks'], source_tap['point_name2id']

    with open(target_point_path, 'r') as fr:
        target_point_dict = json.load(fr)

    for k, v in tqdm(source_point_dict.items()):
        point_idx = point_name2id[k]

        if k in target_point_dict:  # modify the point position
            # clear pre_track
            pred_tracks[:, point_idx, :] = torch.tensor([-1, -1])

            source_y, source_x = v
            target_y, target_x = target_point_dict[k]

            source_xyt_foreground = torch.tensor([norm_Scoord_func(source_x), norm_Scoord_func(source_y), norm_Tcoord_func(keyframe_timestep)]).float().to(device)
            source_uv_foreground = FG_UV_Mapping(source_xyt_foreground.unsqueeze(0))

            # keyframe point -> cononical space -> inverse mapping to all frames: base coordinate
            T = torch.arange(number_of_frames).unsqueeze(-1).to(device)
            source_uv_foreground = repeat(source_uv_foreground, 'b c -> (b t) c', t=T.shape[0])
            source_uvt_foreground = torch.cat([source_uv_foreground, norm_Tcoord_func(T)], dim=-1)
            source_xyt_pred = FG_UV_Mapping_Inverse(source_uvt_foreground)

            # delta coordinate
            dx_dy_allT = propogate_point(source_x, source_y, target_x, target_y, keyframe_timestep, number_of_frames, FG_UV_Mapping, FG_UV_Mapping_Inverse, norm_Scoord_func, norm_Tcoord_func, device)
            warp_xy_pred = source_xyt_pred[:, :2] + dx_dy_allT  # [timestep, 2]

            alpha_pred = 0.5 * (F_Alpha(source_xyt_pred) + 1.0)

            for frame_id in range(number_of_frames):
                if alpha_pred[frame_id] > 0.5:
                    x_pred, y_pred = warp_xy_pred[frame_id]
                    x_pred = torch.round((x_pred + 1) / 2 * larger_dim)
                    y_pred = torch.round((y_pred + 1) / 2 * larger_dim)
                    pred_tracks[frame_id, point_idx, 0], pred_tracks[frame_id, point_idx, 1] = x_pred, y_pred

    source_tap['pred_tracks'] = pred_tracks.cpu()
    return source_tap

def process_displacement_propagation(atlas_config_path, atlas_model_path, source_tap_path, source_point_path, target_point_path):
    # step 1: init atlas model
    atlas_config = OmegaConf.to_container(OmegaConf.load(atlas_config_path), resolve=True)
    FG_UV_Mapping, FG_UV_Mapping_Inverse, F_Alpha = init_atlas_model(atlas_config, atlas_model_path)

    # step 2: propagation source point
    data_dict = load_input_data(atlas_config['datasets'])

    number_of_frames = data_dict['video_frames'].shape[-1]

    larger_dim = np.maximum(data_dict['video_frames'].shape[0], data_dict['video_frames'].shape[1])
    norm_Scoord_func = lambda x: x / (larger_dim / 2) - 1  # noqa
    norm_Tcoord_func = lambda x: x / (number_of_frames / 2) - 1  # noqa

    target_tap = propagate_point_sequence(
        source_point_path=source_point_path, source_tap_path=source_tap_path, target_point_path=target_point_path,
        FG_UV_Mapping=FG_UV_Mapping, FG_UV_Mapping_Inverse=FG_UV_Mapping_Inverse, F_Alpha=F_Alpha,
        larger_dim=larger_dim, number_of_frames=number_of_frames, norm_Scoord_func=norm_Scoord_func, norm_Tcoord_func=norm_Tcoord_func)
    return target_tap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas_config_path', type=str, default='experiments/pretrained_models/atlas_model/animal_atlas/4032_4_atlas_swan_inv_fp32/4032_4_atlas_swan_inv_fp32.yml')
    parser.add_argument('--atlas_model_path', type=str, default='experiments/pretrained_models/atlas_model/animal_atlas/4032_4_atlas_swan_inv_fp32/models/models_40000.pth')
    parser.add_argument('--source_point_path', type=str, default='datasets/paper_evaluation/animal/blackswan/annotation/00000.json')
    parser.add_argument('--source_tap_path', type=str, default='datasets/paper_evaluation/animal/blackswan/annotation/TAP.pth')
    parser.add_argument('--target_point_path', type=str, default='datasets/paper_evaluation/animal/blackswan/annotation/edit_point_try2/00000_catA.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_tap = process_displacement_propagation(
        atlas_config_path = args.atlas_config_path,
        atlas_model_path = args.atlas_model_path,
        source_tap_path = args.source_tap_path,
        source_point_path = args.source_point_path,
        target_point_path = args.target_point_path
    )

    # visualize/save the propagated point sequence
    save_dir = os.path.dirname(args.target_point_path)
    save_suffix = osp.splitext(osp.basename(args.target_point_path))[0]
    tap_save_path = os.path.join(save_dir, f'TAP_{save_suffix}.pth')
    torch.save(target_tap, tap_save_path)
    print(f'save to {tap_save_path}')

    tap_vis_save_dir = os.path.join(save_dir, f'TAP_{save_suffix}')
    os.makedirs(tap_vis_save_dir, exist_ok=True)
    atlas_config = OmegaConf.to_container(OmegaConf.load(args.atlas_config_path), resolve=True)
    visualize_point_sequence(atlas_config['datasets']['frame_path'], target_tap, save_dir=tap_vis_save_dir)
