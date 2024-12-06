import argparse
import json
import os
import os.path as osp
import random

import torch
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from videoswap.data import build_dataset
from videoswap.models import build_model
from videoswap.pipelines import build_pipeline
from videoswap.utils.edlora_util import revise_edlora_unet_attention_forward
from videoswap.utils.logger import dict2str, set_path_logger
from videoswap.utils.vis_util import save_video_to_dir


def test(root_path, opt, opt_path):
    # load config

    # set accelerator, mix-precision set in the environment by "accelerate config"
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, opt_path, opt, is_train=False)

    # get logger
    logger = get_logger('videoswap', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)

    logger.info(dict2str(opt))

    # If passed along, set the seed now.
    if opt.get('manual_seed') is None:
        opt['manual_seed'] = random.randint(1, 10000)
    set_seed(opt['manual_seed'])

    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
        print('enable float16 in the training and testing')
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    unet_type = opt['models']['unet'].pop('type')
    if unet_type == 'AnimateDiffUNet3DModel':
        inference_config_path = opt['models']['unet'].pop('inference_config_path')
        unet = build_model(unet_type).from_pretrained_2d(
            opt['path']['pretrained_model_path'],
            subfolder='unet',
            unet_additional_kwargs=OmegaConf.to_container(OmegaConf.load(inference_config_path).unet_additional_kwargs),
        )
        if opt['models']['unet'].get('motion_module_path'):
            motion_module_path = opt['models']['unet'].pop('motion_module_path')
            motion_module_state_dict = torch.load(motion_module_path, map_location='cpu')
            motion_module_state_dict = {k.replace('.pos_encoder','.processor.pos_encoder'):v for k, v in motion_module_state_dict.items()}
            missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
    else:
        raise NotImplementedError

    adapter_type = opt['models']['adapter'].pop('type')
    t2i_adapter = build_model(adapter_type)(**OmegaConf.to_container(OmegaConf.load(opt['models']['adapter']['model_config_path'])))
    t2i_adapter.load_state_dict(torch.load(opt['path']['pretrained_adapter_path']))
    t2i_adapter = t2i_adapter.to(dtype=weight_dtype)

    val_pipeline = build_pipeline(opt['val']['val_pipeline']).from_pretrained(
        opt['path']['pretrained_model_path'],
        unet=unet.to(dtype=weight_dtype),
        adapter=t2i_adapter,
        scheduler=DDIMScheduler.from_pretrained(opt['path']['pretrained_model_path'], subfolder='scheduler',),
        torch_dtype=weight_dtype
    ).to('cuda')
    val_pipeline.enable_vae_slicing()

    if os.path.exists(os.path.join(opt['path']['pretrained_model_path'], 'new_concept_cfg.json')):
        with open(os.path.join(opt['path']['pretrained_model_path'], 'new_concept_cfg.json'), 'r') as json_file:
            new_concept_cfg = json.load(json_file)
        revise_edlora_unet_attention_forward(val_pipeline.unet)
        val_pipeline.set_new_concept_cfg(new_concept_cfg)

    val_pipeline.scheduler.set_timesteps(opt['val']['editing_config']['num_inference_steps'])
    val_pipeline.enable_vae_slicing()

    # 2. load data
    dataset_opt = opt['datasets']
    dataset_type = dataset_opt.pop('type')
    test_dataset = build_dataset(dataset_type)(dataset_opt)
    source_frames = test_dataset.get_frames()

    if t2i_adapter is not None:
        t2i_adapter.eval()
        source_conditions = test_dataset.get_conditions()
    else:
        source_conditions = None

    edited_results = val_pipeline.validation(
        source_video=source_frames,
        source_conditions=source_conditions,
        source_prompt=opt['datasets']['prompt'],
        editing_config=opt['val']['editing_config'],
        train_dataset=test_dataset,
        save_dir=opt['path']['visualization']
    )

    save_video_to_dir(source_frames, save_dir=os.path.join(opt['path']['visualization'], 'source'), save_suffix='source', save_type=opt['val'].get('save_type', 'frame_gif'), fps=opt['val'].get('fps', 8))
    for key, edit_video in edited_results.items():
        if 'frame' not in opt['val'].get('save_type', 'frame_gif'):
            save_dir = os.path.join(opt['path']['visualization'])
        else:
            save_dir = os.path.join(opt['path']['visualization'], key)

        if opt['val'].get('use_suffix', False):
            save_suffix = f"{key}_{opt['name']}"
        else:
            save_suffix = f'{key}'

        save_video_to_dir(edit_video, save_dir=save_dir, save_suffix=save_suffix, save_type=opt['val'].get('save_type', 'frame_gif'), fps=opt['val'].get('fps', 8))

    return save_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/test/test_jeep_posche.yaml')
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    test(root_path, opt, args.opt)
