import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from videoswap.utils.dift_util import DIFT_Demo, SDFeaturizer
from videoswap.utils.vis_util import get_openpose_name2id, visualize_point_sequence

sys.path.insert(0, './thirdparty/co-tracker')
from cotracker.predictor import CoTrackerPredictor  # noqa


def read_frames_from_folder(frames_folder):
    frames = []
    for filename in sorted(os.listdir(frames_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_path = os.path.join(frames_folder, filename)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))
    return np.stack(frames)


def progagate_human_keypoint(frame_dir):
    openpose_name2id = get_openpose_name2id()
    openpose_id2name = {v: k for k, v in openpose_name2id.items()}

    model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    frame_list = sorted(Path(frame_dir).iterdir())

    ignore_point = ['Neck', 'Left Ear', 'Right Ear', 'Background']

    point_name2id = {}
    all_frame_points = []
    for frame_idx, frame_path in enumerate(tqdm(frame_list)):
        frame = Image.open(frame_path).convert('RGB')
        W, H = frame.size

        input_image = np.array(frame, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, resolution=512)

        poses = model.detect_poses(input_image)[0].body.keypoints  # assume one person

        for point_idx, keypoint in enumerate(poses):
            if keypoint is not None and openpose_id2name[point_idx] not in ignore_point:
                if openpose_id2name[point_idx] not in point_name2id:
                    point_name2id[openpose_id2name[point_idx]] = len(point_name2id)
        all_frame_points.append(poses)

    pred_tracks = torch.zeros((len(frame_list), len(point_name2id.keys()), 2))
    for frame_idx, frame_path in enumerate(tqdm(frame_list)):
        for point_idx, keypoint in enumerate(all_frame_points[frame_idx]):
            if openpose_id2name[point_idx] in point_name2id.keys():
                if keypoint is not None:
                    x, y = W * keypoint.x, H * keypoint.y
                    filtered_point_idx = point_name2id[openpose_id2name[point_idx]]
                    pred_tracks[frame_idx, filtered_point_idx, 0] = x
                    pred_tracks[frame_idx, filtered_point_idx, 1] = y
                else:
                    filtered_point_idx = point_name2id[openpose_id2name[point_idx]]
                    pred_tracks[frame_idx, filtered_point_idx, 0] = -1
                    pred_tracks[frame_idx, filtered_point_idx, 1] = -1

    tap_dict = {'pred_tracks': pred_tracks.cpu(), 'point_name2id': point_name2id}
    return tap_dict


def progagate_general_keypoint(frame_dir, annotation_path):
    # read frames
    video = read_frames_from_folder(frame_dir)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    # init model
    tracker_model = CoTrackerPredictor(checkpoint='thirdparty/co-tracker/checkpoints/cotracker_stride_4_wind_8.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker_model = tracker_model.to(device)
    video = video.to(device)

    timestep = float(osp.splitext(osp.basename(annotation_path))[0])  # keyframe annotation should be frameid.json, e.g., 00010.json
    with open(annotation_path, 'r') as fr:
        json_dict = json.load(fr)

    point_list = []
    point_name2id = {}

    for idx, (key, value) in enumerate(json_dict.items()):
        value = [timestep] + value[::-1]  # [timestep, x, y]
        point_list.append(value)
        point_name2id[key] = idx

    queries = torch.tensor(point_list).to(device)
    pred_tracks, pre_vis = tracker_model(  # the pred visibility is not accurate,
        video,
        queries=queries[None],
        backward_tracking=True,
    )
    pred_tracks = pred_tracks.squeeze(0)
    tap_dict = {'pred_tracks': pred_tracks.cpu(), 'point_name2id': point_name2id}
    return tap_dict


def extract_dift_feature(image, category, dift_model):
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
    prompt = f'photo of a {category}'
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    dift_feature = dift_model.forward(img_tensor, prompt=prompt, ensemble_size=8)
    return dift_feature


def extract_point_embedding(tap_dict, frame_dir, keyframe_annotation_path, model_id, subject_category, is_human=True):
    if is_human:
        confidence_threshold = 0.7
        num_point = tap_dict['pred_tracks'].shape[1]
        init_embedding = torch.zeros((num_point, 1280))
        init_count = torch.zeros(num_point)

        dift_model = SDFeaturizer(sd_id=model_id)
        for frame_path in tqdm(sorted(Path(frame_dir).iterdir())):
            img = Image.open(frame_path).convert('RGB')
            x, y = img.size
            img_dift = extract_dift_feature(img, category=subject_category, dift_model=dift_model)
            img_dift = nn.Upsample(size=(y, x), mode='bilinear')(img_dift)

            frame_id = osp.splitext(osp.basename(frame_path))[0]
            frame_point = tap_dict['pred_tracks'][int(frame_id)]

            for point_idx, point in enumerate(frame_point):
                point_x, point_y = int(np.round(point[0])), int(np.round(point[1]))

                if point_x >= 0 and point_y >= 0:
                    point_embedding = img_dift[0, :, point_y, point_x]
                    init_embedding[point_idx] += point_embedding.cpu()
                    init_count[point_idx] += 1

        for point_idx in range(init_count.shape[0]):
            if init_count[point_idx] != 0:
                init_embedding[point_idx] /= init_count[point_idx]

        tap_dict['point_embedding'] = init_embedding
    else:
        confidence_threshold = 0.35
        dift_model = SDFeaturizer(sd_id=model_id)

        keyframe_id = osp.splitext(osp.basename(keyframe_annotation_path))[0]
        keyframe_image = Image.open(os.path.join(frame_dir, f'{keyframe_id}.jpg')).convert('RGB')
        keyframe_dift = extract_dift_feature(keyframe_image, category=subject_category, dift_model=dift_model)

        width, height = keyframe_image.size
        dift_demo = DIFT_Demo(source_img=keyframe_image, source_dift=keyframe_dift, source_img_size=[height, width])

        keyframe_point = tap_dict['pred_tracks'][int(keyframe_id)]
        num_point = keyframe_point.shape[0]
        init_embedding = torch.zeros((num_point, 1280))
        init_count = torch.zeros(num_point)

        for frame_path in tqdm(Path(frame_dir).iterdir()):
            target_img = Image.open(frame_path).convert('RGB')
            target_dift = extract_dift_feature(target_img, category=subject_category, dift_model=dift_model)
            target_frame_id = int(osp.splitext(osp.basename(frame_path))[0])
            width, height = target_img.size

            for point_idx, point in enumerate(keyframe_point):
                src_x, src_y = point.numpy()
                src_x, src_y = np.round(src_x), np.round(src_y)

                tgt_x, tgt_y = tap_dict['pred_tracks'][int(target_frame_id)][point_idx]
                tgt_x, tgt_y = np.round(tgt_x), np.round(tgt_y)

                if tgt_x >= width or tgt_y >= height:
                    tap_dict['pred_tracks'][int(target_frame_id)][point_idx] = torch.tensor([-1, -1])
                    continue

                point_feat, confidence, _ = dift_demo.query(
                    target_img, target_dift=target_dift, target_img_size=[height, width], query_point=[src_y, src_x], target_point=[tgt_y, tgt_x], visualize=False)

                if confidence >= confidence_threshold:
                    init_embedding[point_idx] += point_feat.cpu()
                    init_count[point_idx] += 1
                else:
                    tap_dict['pred_tracks'][int(target_frame_id)][point_idx] = torch.tensor([-1, -1])

        print('filtered point count:', init_count)

        for point_idx in range(init_count.shape[0]):
            if init_count[point_idx] != 0:
                init_embedding[point_idx] /= init_count[point_idx]

        tap_dict['point_embedding'] = init_embedding
    return tap_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', type=str, default='datasets/paper_evaluation/object/car_turn/frames')
    parser.add_argument('--is_human', type=bool, default=False)
    parser.add_argument('--keyframe_annotation_path', type=str, default='datasets/paper_evaluation/object/car_turn/annotation_v4/00035.json')
    parser.add_argument('--save_dir', type=str, default='datasets/paper_evaluation/object/car_turn/annotation_v4')
    parser.add_argument('--model_id', type=str, default='experiments/pretrained_models/chilloutmix')
    parser.add_argument('--subject_category', type=str, default='car')
    args = parser.parse_args()

    # step1: point coordinate extraction: propagate keyframe control point to others
    if args.is_human:
        tap_dict = progagate_human_keypoint(args.frame_dir)
    else:
        tap_dict = progagate_general_keypoint(args.frame_dir, args.keyframe_annotation_path)

    # step2: point embedding extraction (filter out wrong point coordinate here)
    tap_dict = extract_point_embedding(
        tap_dict, args.frame_dir, args.keyframe_annotation_path, args.model_id, args.subject_category, is_human=args.is_human)

    # visualize point sequence
    visualize_point_sequence(args.frame_dir, tap_dict, save_dir=os.path.join(args.save_dir, 'tap_vis'))
    torch.save(tap_dict, os.path.join(args.save_dir, 'TAP.pth'))
