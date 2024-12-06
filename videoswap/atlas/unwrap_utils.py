from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm


def compute_consistency(flow12, flow21):
    wflow21 = warp_flow(flow21, flow12)
    diff = flow12 + wflow21
    diff = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) ** .5
    return diff


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def get_consistency_mask(optical_flow, optical_flow_reverse):
    mask_flow = compute_consistency(optical_flow, optical_flow_reverse) < 1.0
    mask_flow_reverse = compute_consistency(optical_flow_reverse, optical_flow) < 1.0
    return mask_flow, mask_flow_reverse


def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[0:2]
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= newh / oldh
    flow[:, :, 1] *= neww / oldw
    return flow


def load_input_data(datasets_opt):
    frame_files = sorted(Path(datasets_opt['frame_path']).iterdir())
    number_of_frames = np.minimum(datasets_opt['max_frames'], len(frame_files))

    video_frames = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], 3, number_of_frames))
    video_frames_dx = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], 3, number_of_frames))
    video_frames_dy = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], 3, number_of_frames))

    mask_frames = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], number_of_frames))

    optical_flows = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], 2, number_of_frames, 1))
    optical_flows_mask = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], number_of_frames, 1))
    optical_flows_reverse = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], 2, number_of_frames, 1))
    optical_flows_reverse_mask = torch.zeros((datasets_opt['res_y'], datasets_opt['res_x'], number_of_frames, 1))

    # read masks
    mask_files = sorted(Path(datasets_opt['mask_path']).iterdir())
    flow_files = sorted(Path(datasets_opt['flow_path']).iterdir())

    for idx in range(number_of_frames):
        image = np.array(Image.open(frame_files[idx]).convert('RGB')).astype(np.float64) / 255.
        mask = np.array(Image.open(mask_files[idx]).convert('L')).astype(np.float64) / 255.

        image = cv2.resize(image, (datasets_opt['res_x'], datasets_opt['res_y']))
        mask = cv2.resize(mask, (datasets_opt['res_x'], datasets_opt['res_y']), cv2.INTER_NEAREST)

        video_frames[:, :, :, idx] = torch.from_numpy(image)
        mask_frames[:, :, idx] = torch.from_numpy(mask)

        video_frames_dy[:-1, :, :, idx] = video_frames[1:, :, :, idx] - video_frames[:-1, :, :, idx]
        video_frames_dx[:, :-1, :, idx] = video_frames[:, 1:, :, idx] - video_frames[:, :-1, :, idx]

        if idx < number_of_frames - 1:
            flow = np.load(flow_files[idx])
            forward_flow, backward_flow = flow[0], flow[1]
            if flow.shape[1] != datasets_opt['res_y'] or flow.shape[2] != datasets_opt['res_x']:
                forward_flow = resize_flow(forward_flow, newh=datasets_opt['res_y'], neww=datasets_opt['res_x'])
                backward_flow = resize_flow(backward_flow, newh=datasets_opt['res_y'], neww=datasets_opt['res_x'])

            mask_flow, mask_flow_reverse = get_consistency_mask(forward_flow, backward_flow)
            optical_flows[:, :, :, idx, 0] = torch.from_numpy(forward_flow)
            optical_flows_reverse[:, :, :, idx + 1, 0] = torch.from_numpy(backward_flow)
            if datasets_opt['filter_optical_flow']:
                optical_flows_mask[:, :, idx, 0] = torch.from_numpy(mask_flow)
                optical_flows_reverse_mask[:, :, idx + 1, 0] = torch.from_numpy(mask_flow_reverse)
            else:
                optical_flows_mask[:, :, idx, 0] = torch.ones_like(mask_flow)
                optical_flows_reverse_mask[:, :, idx + 1, 0] = torch.ones_like(mask_flow_reverse)

    data_dict = {
        'video_frames': video_frames,
        'mask_frames': mask_frames,
        'video_frames_dx': video_frames_dx,
        'video_frames_dy': video_frames_dy,
        'optical_flows': optical_flows,
        'optical_flows_mask': optical_flows_mask,
        'optical_flows_reverse': optical_flows_reverse,
        'optical_flows_reverse_mask': optical_flows_reverse_mask
    }
    return data_dict


def get_tuples(number_of_frames, video_frames):
    # video_frames shape: (resy, resx, 3, num_frames), mask_frames shape: (resy, resx, num_frames)
    jif_all = []
    for f in range(number_of_frames):
        mask = (video_frames[:, :, :, f] > -1).any(dim=2)
        relis, reljs = torch.where(mask > 0.5)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1)


# See explanation in the paper, appendix A (Second paragraph)
def pretrain_mapping(model_UV_mapping, uv_mapping_scale, resx, resy, frames_num,
                     norm_Scoord_func, norm_Tcoord_func, device, pretrain_iters=100):

    optimizer_mapping = optim.Adam(model_UV_mapping.parameters(), lr=0.0001)
    for _ in tqdm(range(pretrain_iters), desc='pretrain UV mapping'):
        loss_sum = 0.0
        for f in range(frames_num):
            i_s_int = torch.randint(resy, (np.int64(10000), 1))
            j_s_int = torch.randint(resx, (np.int64(10000), 1))

            i_s = norm_Scoord_func(i_s_int)
            j_s = norm_Scoord_func(j_s_int)

            xyt = torch.cat((j_s, i_s, norm_Tcoord_func(f) * torch.ones_like(i_s)), dim=1).to(device)
            uv_temp = model_UV_mapping(xyt)

            model_UV_mapping.zero_grad()

            loss = (xyt[:, :2] * uv_mapping_scale - uv_temp).norm(dim=1).mean()
            loss.backward()
            optimizer_mapping.step()
            loss_sum += loss.item()

    return model_UV_mapping, loss_sum / frames_num


def save_mask_flow(optical_flows_mask, video_frames, results_folder):
    for j in range(optical_flows_mask.shape[3]):

        filter_flow_0 = imageio.get_writer('%s/filter_flow_%d.mp4' % (results_folder, j), fps=10)
        for i in range(video_frames.shape[3]):
            if torch.where(optical_flows_mask[:, :, i, j] == 1)[0].shape[0] == 0:
                continue
            cur_frame = video_frames[:, :, :, i].clone()
            # Put red color where mask=0.
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0], torch.where(optical_flows_mask[:, :, i, j] == 0)[1], 0] = 1
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0], torch.where(optical_flows_mask[:, :, i, j] == 0)[1], 1] = 0
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0], torch.where(optical_flows_mask[:, :, i, j] == 0)[1], 2] = 0

            filter_flow_0.append_data((cur_frame.numpy() * 255).astype(np.uint8))

        filter_flow_0.close()

    # save the video in the working resolution
    input_video = imageio.get_writer('%s/input_video.mp4' % (results_folder), fps=10)
    for i in range(video_frames.shape[3]):
        cur_frame = video_frames[:, :, :, i].clone()
        input_video.append_data((cur_frame.numpy() * 255).astype(np.uint8))

    input_video.close()
