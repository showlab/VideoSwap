import os
import os.path as osp
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image, ImageDraw


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, 'b c t h w -> t b c h w')
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=1000 / fps)


def save_grid_images(image_list, n_rows=3, n_cols=4):
    # 计算画布的宽度和高度
    canvas_width = image_list[0].width * n_cols
    canvas_height = image_list[0].height * n_rows

    # 创建一个空白的画布
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    # 将图像填充到画布上
    for i, image in enumerate(image_list):
        # 计算图像在网格中的位置
        row = i // n_cols
        col = i % n_cols

        # 计算图像在画布中的位置
        x = col * image.width
        y = row * image.height

        # 将图像粘贴到画布上
        canvas.paste(image, (x, y))

    # 保存画布
    return canvas


def save_images_as_gif(images: List[Image.Image],
                       save_path: str,
                       fps=5) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=int(1000 / fps),
    )


def save_video_to_dir(edit_video, save_dir, save_suffix, save_type='frame', fps=8):
    os.makedirs(save_dir, exist_ok=True)

    save_type_list = save_type.split('_')

    # save frame
    if 'frame' in save_type_list:
        frame_save_dir = os.path.join(save_dir, 'frames')
        os.makedirs(frame_save_dir, exist_ok=True)
        for idx, img in enumerate(edit_video):
            img.save(os.path.join(frame_save_dir, f'{idx:05d}_{save_suffix}.jpg'))

    # save to gif
    if 'gif' in save_type_list:
        gif_save_path = os.path.join(save_dir, f'{save_suffix}.gif')
        save_images_as_gif(edit_video, gif_save_path, fps=fps)

    # save to video
    if 'video' in save_type_list:
        video_save_path = os.path.join(save_dir, f'{save_suffix}.mp4')
        export_to_video(edit_video, video_save_path, fps=fps)


def export_to_video(video_frames: List[Image.Image], output_video_path: str, fps=8) -> str:
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    video_writer = imageio.get_writer(output_video_path, fps=fps)

    # Write each image to the video
    for img in video_frames:
        video_writer.append_data(np.array(img))

    # Close the video writer
    video_writer.close()
    return output_video_path


def visualize_point_sequence(frame_dir, TAP_path, save_dir=None, selected_point=None):
    
    if isinstance(TAP_path, dict):
        TAP_dict = TAP_path
    else:
        TAP_dict = torch.load(TAP_path)

    pred_tracks = TAP_dict['pred_tracks']
    point_name2id = TAP_dict['point_name2id']
    
    select_point_idx = None
    if selected_point is not None:
        select_point_idx = []
        for name in selected_point:
            select_point_idx.append(point_name2id[name])
    
    frame_list = sorted(Path(frame_dir).iterdir())
    point_nums = pred_tracks.shape[1]

    result_frames = []

    for idx, image_path in enumerate(frame_list):

        all_points = []
        all_colors = []

        image = Image.open(image_path)
        pred_track_in_frame = pred_tracks[idx]

        for point_idx in range(point_nums):
            if select_point_idx is not None and point_idx not in select_point_idx:
                continue
            x, y = pred_track_in_frame[point_idx]
            if x != -1 and y != -1:
                all_points.append(pred_track_in_frame[point_idx])
                all_colors.append((0, 255, 0))

        # 在图像上标记点
        draw = ImageDraw.Draw(image)
        radius = 3
        for point, color in zip(all_points, all_colors):
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        if save_dir is not None:
            basename = osp.splitext(osp.basename(image_path))[0]
            os.makedirs(save_dir, exist_ok=True)
            image.save(f'{save_dir}/{basename}.jpg')
        
        result_frames.append(image)
    return result_frames


def get_openpose_name2id():
    point_name2id = {
        'Nose': 0, 'Neck': 1, 'Right Shoulder': 2, 'Right Elbow': 3, 'Right Wrist': 4,
        'Left Shoulder': 5, 'Left Elbow': 6, 'Left Wrist': 7, 'Right Hip': 8, 'Right Knee': 9,
        'Right Ankle': 10, 'Left Hip': 11, 'Left Knee': 12, 'Left Ankle': 13, 'Right Eye': 14,
        'Left Eye': 15, 'Right Ear': 16, 'Left Ear': 17, 'Background': 18
    }
    return point_name2id
