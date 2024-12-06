import argparse
import copy
import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

from videoswap.atlas.evaluate import evaluate_model
from videoswap.atlas.implicit_neural_networks import IMLP_Hash, count_parameters
from videoswap.atlas.loss_utils import (get_gradient_loss, get_optical_flow_alpha_loss, get_optical_flow_loss,
                                        get_rigidity_loss)
from videoswap.atlas.unwrap_utils import get_tuples, load_input_data, pretrain_mapping, save_mask_flow
from videoswap.utils.logger import MessageLogger, dict2str, reduce_loss_dict, set_path_logger


def train(root_path, args):
    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator, mix-precision set in the environment by "accelerate config"
    accelerator = Accelerator(
        mixed_precision=opt['mixed_precision'],
    )

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, args.opt, opt, is_train=True)

    # get logger
    logger = get_logger('videoswap', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)

    logger.info(dict2str(opt))

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is None:
        opt['manual_seed'] = random.randint(1, 10000)
    set_seed(opt['manual_seed'])

    # build model
    FG_UV_Mapping = IMLP_Hash(**opt['models']['FG_UV_Mapping']).to(accelerator.device)
    logger.info(f'FG_UV_Mapping has {count_parameters(FG_UV_Mapping)} params')
    BG_UV_Mapping = IMLP_Hash(**opt['models']['BG_UV_Mapping']).to(accelerator.device)
    logger.info(f'BG_UV_Mapping has {count_parameters(BG_UV_Mapping)} params')
    F_Alpha = IMLP_Hash(**opt['models']['F_Alpha']).to(accelerator.device)
    logger.info(f'F_Alpha has {count_parameters(F_Alpha)} params')
    F_Atlas = IMLP_Hash(**opt['models']['F_Atlas']).to(accelerator.device)
    logger.info(f'F_Atlas has {count_parameters(F_Atlas)} params')

    if 'FG_UV_Mapping_Inverse' in opt['models']:
        FG_UV_Mapping_Inverse = IMLP_Hash(**opt['models']['FG_UV_Mapping_Inverse']).to(accelerator.device)
        logger.info(f'FG_UV_Mapping_Inverse has {count_parameters(FG_UV_Mapping_Inverse)} params')
    else:
        FG_UV_Mapping_Inverse = None

    data_dict = load_input_data(opt['datasets'])

    # save a video showing the masked part of the forward optical flow:s
    save_mask_flow(data_dict['optical_flows_mask'], data_dict['video_frames'], opt['path']['visualization'])

    params_to_optimize = [
        {'params': list(FG_UV_Mapping.parameters())},
        {'params': list(BG_UV_Mapping.parameters())},
        {'params': list(F_Alpha.parameters())},
        {'params': list(F_Atlas.parameters())}
    ]
    optim_type = opt['train']['optimizer'].pop('type')
    if optim_type == 'Adam':
        optimizer_all = optim.Adam(params_to_optimize, **opt['train']['optimizer'])
    else:
        raise NotImplementedError

    if FG_UV_Mapping_Inverse is not None:
        inverse_params_to_optimize = [
            {'params': list(FG_UV_Mapping_Inverse.parameters())},
        ]
        if optim_type == 'Adam':
            optimizer_inverse = optim.Adam(inverse_params_to_optimize, **opt['train']['optimizer'])
        else:
            raise NotImplementedError

        FG_UV_Mapping, FG_UV_Mapping_Inverse, BG_UV_Mapping, F_Alpha, F_Atlas, optimizer_all, optimizer_inverse = accelerator.prepare(FG_UV_Mapping, FG_UV_Mapping_Inverse, BG_UV_Mapping, F_Alpha, F_Atlas, optimizer_all, optimizer_inverse)
    else:
        optimizer_inverse = None
        FG_UV_Mapping, BG_UV_Mapping, F_Alpha, F_Atlas, optimizer_all = accelerator.prepare(FG_UV_Mapping, BG_UV_Mapping, F_Alpha, F_Atlas, optimizer_all)

    number_of_frames = data_dict['video_frames'].shape[-1]
    larger_dim = np.maximum(data_dict['video_frames'].shape[0], data_dict['video_frames'].shape[1])
    norm_Scoord_func = lambda x: x / (larger_dim / 2) - 1  # noqa
    norm_Tcoord_func = lambda x: x / (number_of_frames / 2) - 1  # noqa

    if opt['train']['pretrain_UV_mapping_iter'] > 0:
        FG_UV_Mapping, loss_pretrain = pretrain_mapping(
            FG_UV_Mapping, uv_mapping_scale=opt['train']['uv_mapping_scale'],
            resx=opt['datasets']['res_x'], resy=opt['datasets']['res_y'], frames_num=number_of_frames,
            norm_Scoord_func=norm_Scoord_func, norm_Tcoord_func=norm_Tcoord_func,
            device=accelerator.device, pretrain_iters=opt['train']['pretrain_UV_mapping_iter'])
        logger.info(f'Finish pretrain FG_UV_Mapping with final loss: {loss_pretrain:.4f}')
        BG_UV_Mapping, loss_pretrain = pretrain_mapping(
            BG_UV_Mapping, uv_mapping_scale=opt['train']['uv_mapping_scale'],
            resx=opt['datasets']['res_x'], resy=opt['datasets']['res_y'], frames_num=number_of_frames,
            norm_Scoord_func=norm_Scoord_func, norm_Tcoord_func=norm_Tcoord_func,
            device=accelerator.device, pretrain_iters=opt['train']['pretrain_UV_mapping_iter'])
        logger.info(f'Finish pretrain BG_UV_Mapping with final loss: {loss_pretrain:.4f}')

    jif_all = get_tuples(number_of_frames, data_dict['video_frames'])

    train_opt = opt['train']
    # Start training!

    global_step = 0
    msg_logger = MessageLogger(opt, global_step)

    while global_step < train_opt['total_iter']:

        FG_UV_Mapping.train()
        BG_UV_Mapping.train()
        F_Alpha.train()
        F_Atlas.train()

        # randomly choose indices for the current batch
        inds_foreground = torch.randint(jif_all.shape[1], (np.int64(opt['datasets']['sample_batch_size'] * 1.0), 1))

        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)

        rgb_current = data_dict['video_frames'][jif_current[1, :], jif_current[0, :], :, jif_current[2, :]].squeeze(1).to(accelerator.device)

        # the correct alpha according to the precomputed maskrcnn
        alpha_gt = data_dict['mask_frames'][jif_current[1, :], jif_current[0, :], jif_current[2, :]].squeeze(1).to(accelerator.device).unsqueeze(-1)

        # normalize coordinates to be in [-1,1]
        xyt_current = torch.cat([norm_Scoord_func(jif_current[0, :]), norm_Scoord_func(jif_current[1, :]), norm_Tcoord_func(jif_current[2, :])], dim=1).to(accelerator.device)

        # get the atlas UV coordinates from the two mapping networks;
        uv_fg = FG_UV_Mapping(xyt_current)
        uv_bg = BG_UV_Mapping(xyt_current)

        # map tanh output of the alpha network to the range (0,1) :
        alpha = 0.5 * (F_Alpha(xyt_current) + 1.0)
        # prevent a situation of alpha=0, or alpha=1 (for the BCE loss that uses log(alpha),log(1-alpha) below)
        alpha = alpha * 0.99
        alpha = alpha + 0.001

        # Sample atlas values. Foreground colors are sampled from [0,1]x[0,1] and background colors are sampled from [-1,0]x[-1,0]
        # Note that the original [u,v] coorinates are in [-1,1]x[-1,1] for both networks
        rgb_output_fg = (F_Atlas(uv_fg * 0.5 + 0.5) + 1.0) * 0.5
        rgb_output_bg = (F_Atlas(uv_bg * 0.5 - 0.5) + 1.0) * 0.5
        # Reconstruct final colors from the two layers (using alpha)
        rgb_output = rgb_output_fg * alpha + rgb_output_bg * (1.0 - alpha)

        # ---------------------------------loss calculation-------------------------------------

        loss_total = 0.0
        loss_dict = {}

        # reconstruction loss
        loss_cfg = train_opt['loss_cfg']

        gradient_loss = get_gradient_loss(
            data_dict['video_frames_dx'], data_dict['video_frames_dy'], jif_current,
            FG_UV_Mapping, BG_UV_Mapping, F_Atlas, F_Alpha,
            rgb_output, norm_Scoord_func, norm_Tcoord_func, device=accelerator.device
        )
        loss_dict['gradient_loss'] = gradient_loss
        loss_total += loss_cfg['reconstruction_loss']['gradient_loss_weight'] * gradient_loss

        rgb_loss = (torch.norm(rgb_output - rgb_current, dim=1) ** 2).mean()
        loss_dict['rgb_loss'] = rgb_loss
        loss_total += loss_cfg['reconstruction_loss']['rgb_loss_weight'] * rgb_loss

        if global_step <= train_opt['pretrain_alpha_iter']:
            alpha_loss = torch.mean(-alpha_gt * torch.log(alpha) - (1 - alpha_gt) * torch.log(1 - alpha))
            loss_dict['alpha_loss'] = alpha_loss
            loss_total += loss_cfg['reconstruction_loss']['alpha_loss_weight'] * alpha_loss

        loss_dict['alpha_mean_fg'] = alpha[alpha > 0.5].mean()
        loss_dict['alpha_mean_bg'] = alpha[alpha < 0.5].mean()

        # sparsity loss
        rgb_output_fg_not = rgb_output_fg * (1.0 - alpha)
        rgb_loss_sparsity = (torch.norm(rgb_output_fg_not, dim=1) ** 2).mean()
        loss_dict['sparsity_loss'] = rgb_loss_sparsity
        loss_total += loss_cfg['sparsity_loss']['sparsity_loss_weight'] * rgb_loss_sparsity

        # rigidity loss
        rigidity_loss_fg = get_rigidity_loss(
            jif_current, train_opt['derivative_amount'], larger_dim,
            FG_UV_Mapping, uv_fg, opt['train']['uv_mapping_scale'],
            norm_Scoord_func, norm_Tcoord_func, device=accelerator.device)
        loss_dict['rigidity_loss_fg'] = rigidity_loss_fg
        loss_total += loss_cfg['rigidity_loss']['rigidity_loss_weight'] * rigidity_loss_fg

        rigidity_loss_bg = get_rigidity_loss(
            jif_current, train_opt['derivative_amount'], larger_dim,
            BG_UV_Mapping, uv_bg, opt['train']['uv_mapping_scale'],
            norm_Scoord_func, norm_Tcoord_func, device=accelerator.device)
        loss_dict['rigidity_loss_bg'] = rigidity_loss_bg
        loss_total += loss_cfg['rigidity_loss']['rigidity_loss_weight'] * rigidity_loss_bg

        if global_step <= train_opt['pretrain_global_rigidity_iter']:
            global_rigidity_loss_fg = get_rigidity_loss(
                jif_current, train_opt['global_derivative_amount'], larger_dim,
                FG_UV_Mapping, uv_fg, opt['train']['uv_mapping_scale'],
                norm_Scoord_func, norm_Tcoord_func, device=accelerator.device)
            loss_dict['global_rigidity_loss_fg'] = global_rigidity_loss_fg
            loss_total += loss_cfg['rigidity_loss']['global_rigidity_fg_loss_weight'] * global_rigidity_loss_fg

            global_rigidity_loss_bg = get_rigidity_loss(
                jif_current, train_opt['global_derivative_amount'], larger_dim,
                BG_UV_Mapping, uv_bg, opt['train']['uv_mapping_scale'],
                norm_Scoord_func, norm_Tcoord_func, device=accelerator.device)
            loss_dict['global_rigidity_loss_bg'] = global_rigidity_loss_bg
            loss_total += loss_cfg['rigidity_loss']['global_rigidity_bg_loss_weight'] * global_rigidity_loss_bg

        # flow loss
        flow_loss_fg = get_optical_flow_loss(
            jif_current, uv_fg, data_dict['optical_flows_reverse'], data_dict['optical_flows_reverse_mask'],
            larger_dim, FG_UV_Mapping, data_dict['optical_flows'], data_dict['optical_flows_mask'], opt['train']['uv_mapping_scale'],
            norm_Scoord_func, norm_Tcoord_func, device=accelerator.device, use_alpha=True, alpha=alpha)
        loss_dict['flow_loss_fg'] = flow_loss_fg
        loss_total += loss_cfg['flow_loss']['flow_loss_weight'] * flow_loss_fg

        flow_loss_bg = get_optical_flow_loss(
            jif_current, uv_bg, data_dict['optical_flows_reverse'], data_dict['optical_flows_reverse_mask'],
            larger_dim, BG_UV_Mapping, data_dict['optical_flows'], data_dict['optical_flows_mask'], opt['train']['uv_mapping_scale'],
            norm_Scoord_func, norm_Tcoord_func, device=accelerator.device, use_alpha=True, alpha=1 - alpha)
        loss_dict['flow_loss_bg'] = flow_loss_bg
        loss_total += loss_cfg['flow_loss']['flow_loss_weight'] * flow_loss_bg

        flow_alpha_loss = get_optical_flow_alpha_loss(
            F_Alpha, jif_current, alpha, data_dict['optical_flows_reverse'], data_dict['optical_flows_reverse_mask'],
            norm_Scoord_func, norm_Tcoord_func, data_dict['optical_flows'], data_dict['optical_flows_mask'], device=accelerator.device)
        loss_dict['flow_alpha_loss'] = flow_alpha_loss
        loss_total += loss_cfg['flow_loss']['alpha_flow_loss_weight'] * flow_alpha_loss

        loss_dict['total_loss'] = loss_total

        optimizer_all.zero_grad()
        accelerator.backward(loss_total)
        optimizer_all.step()

        # -------------------------------begin train inverse-------------------------------
        if FG_UV_Mapping_Inverse is not None:
            xyt_fg = xyt_current[alpha_gt.squeeze(1) == 1]
            with torch.no_grad():
                uv_fg = FG_UV_Mapping(xyt_fg)
            inp_uvt_fg = torch.cat([uv_fg, xyt_fg[:, -1:]], dim=-1)
            xyt_pred_fg = FG_UV_Mapping_Inverse(inp_uvt_fg)
            loss = (xyt_pred_fg - xyt_fg).norm(dim=1).mean()
            loss_dict['fg_inv_loss'] = loss
            optimizer_inverse.zero_grad()
            accelerator.backward(loss)
            optimizer_inverse.step()

        log_dict = reduce_loss_dict(accelerator, loss_dict)

        if accelerator.sync_gradients:
            global_step += 1

            if accelerator.is_main_process:
                if global_step % opt['logger']['print_freq'] == 0:
                    log_vars = {'iter': global_step}
                    log_vars.update({'lrs': [optimizer_all.param_groups[0]['lr']]})
                    log_vars.update(log_dict)
                    msg_logger(log_vars)

                if global_step % opt['val']['val_freq'] == 0:
                    save_dir = os.path.join(opt['path']['visualization'], f'Iter_{global_step}')
                    os.makedirs(save_dir, exist_ok=True)

                    psnr = evaluate_model(
                        FG_UV_Mapping, BG_UV_Mapping, F_Atlas, F_Alpha,
                        data_dict['video_frames'], data_dict['mask_frames'], data_dict['optical_flows'], data_dict['optical_flows_mask'],
                        opt['datasets']['res_x'], opt['datasets']['res_y'], number_of_frames, train_opt['derivative_amount'], train_opt['uv_mapping_scale'],
                        save_dir=save_dir, device=accelerator.device
                    )
                    logger.info(f'Validation Reconstruction PSNR: {psnr:.4f}')

                    if FG_UV_Mapping_Inverse is not None:
                        # validate inverse
                        with torch.no_grad():
                            x, y, t = 463, 265, 34
                            validation_point = torch.tensor([[norm_Scoord_func(x), norm_Scoord_func(y), norm_Tcoord_func(t)]]).float().to(accelerator.device)
                            validation_uv = FG_UV_Mapping(validation_point)
                            validation_uvt = torch.cat([validation_uv, validation_point[:, -1:]], dim=-1)
                            validation_inv = FG_UV_Mapping_Inverse(validation_uvt)
                            logger.info(f'inverse pred: {list(validation_inv.cpu().numpy())}, gt: {list(validation_point.cpu().numpy())}')

                        annotation_save_dir = os.path.join(opt['path']['visualization'], f'Iter_{global_step}', 'annotation')
                        annotate_validation(
                            opt['datasets'], FG_UV_Mapping, FG_UV_Mapping_Inverse, F_Alpha,
                            larger_dim, number_of_frames, norm_Scoord_func, norm_Tcoord_func,
                            save_dir=annotation_save_dir, device=accelerator.device)
                        logger.info('Validation Point Propogation!')

                if global_step % opt['logger']['save_checkpoint_freq'] == 0:
                    checkpoint_save_path = os.path.join(opt['path']['models'], f'models_{global_step}.pth')
                    state_dict = {
                        'F_Atlas': F_Atlas.state_dict(),
                        'FG_UV_Mapping': FG_UV_Mapping.state_dict(),
                        'BG_UV_Mapping': BG_UV_Mapping.state_dict(),
                        'F_Alpha': F_Alpha.state_dict()
                    }
                    if FG_UV_Mapping_Inverse is not None:
                        state_dict.update({'FG_UV_Mapping_Inverse': FG_UV_Mapping_Inverse.state_dict()})

                    torch.save(state_dict, checkpoint_save_path)
                    logger.info(f'Save models to {checkpoint_save_path}')


def annotate_validation(dataset_opt, FG_UV_Mapping, FG_UV_Mapping_Inverse, F_Alpha, larger_dim, number_of_frames, norm_Scoord_func, norm_Tcoord_func, save_dir, device):
    annotation_file = dataset_opt['annotation_path']
    with open(annotation_file, 'r') as fr:
        json_dict = json.load(fr)
        timestep = int(osp.splitext(osp.basename(annotation_file))[0])

    empty_json_dict = copy.deepcopy(json_dict)
    for k in empty_json_dict.keys():
        empty_json_dict[k] = []
    json_all_pred = [copy.deepcopy(empty_json_dict) for _ in range(number_of_frames)]

    for k, v in tqdm(json_dict.items()):
        if len(v) != 0:
            h, w = v
            xyt_foreground = torch.tensor([norm_Scoord_func(w), norm_Scoord_func(h), norm_Tcoord_func(timestep)]).float().to(device)
            uv_foreground = FG_UV_Mapping(xyt_foreground.unsqueeze(0))
            T = torch.arange(number_of_frames).unsqueeze(-1).to(device)
            uv_foreground = repeat(uv_foreground, 'b c -> (b t) c', t=T.shape[0])
            uvt_foreground = torch.cat([uv_foreground, norm_Tcoord_func(T)], dim=-1)
            xyt_pred = FG_UV_Mapping_Inverse(uvt_foreground)

            alpha_pred = 0.5 * (F_Alpha(xyt_pred) + 1.0)

            for frame_id in range(number_of_frames):
                if alpha_pred[frame_id] > 0.5:
                    x_pred, y_pred, t_pred = xyt_pred[frame_id]
                    x_pred = torch.round((x_pred + 1) / 2 * larger_dim)
                    y_pred = torch.round((y_pred + 1) / 2 * larger_dim)
                    t_pred = torch.round((t_pred + 1) / 2 * number_of_frames)
                    json_all_pred[frame_id][k] = [int(y_pred), int(x_pred)]

    anno_json_save_dir = os.path.join(save_dir, 'anno_json')
    anno_vis_save_dir = os.path.join(save_dir, 'anno_vis')
    os.makedirs(anno_json_save_dir, exist_ok=True)
    os.makedirs(anno_vis_save_dir, exist_ok=True)

    for idx, json_dict in enumerate(json_all_pred):
        with open(f'{anno_json_save_dir}/{idx:05d}.json', 'w') as fw:
            json.dump(json_dict, fw, indent=4)
    visualize(dataset_opt['frame_path'], anno_json_save_dir, number_of_frames, anno_vis_save_dir)


def visualize(frame_dir, annotation_dir, number_of_frames, save_dir):
    for idx in range(number_of_frames):
        image_path = f'{frame_dir}/{idx:05d}.jpg'
        anno_path = f'{annotation_dir}/{idx:05d}.json'

        image = Image.open(image_path)

        colors = [(0, 255, 0), (51, 153, 255), (255, 128, 0)]  # RGB颜色值

        with open(anno_path, 'r') as fr:
            json_dict = json.load(fr)

        all_points = []
        all_colors = []

        for k, v in json_dict.items():
            if 'Right' in k:
                point = v
                color = colors[0]
            elif 'Left' in k:
                point = v
                color = colors[1]
            else:
                point = v
                color = colors[2]
            if len(point) != 0:
                all_points.append(point)
                all_colors.append(color)

        # 在图像上标记点
        draw = ImageDraw.Draw(image)
        radius = 3  # 圆点的半径
        for point, color in zip(all_points, all_colors):
            y, x = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        image.save(f'{save_dir}/{idx:05d}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train_atlas/train_jeep.yml')
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train(root_path, args)
