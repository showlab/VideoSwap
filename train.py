import argparse
import os
import os.path as osp
import random

import torch
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPTextModel

from videoswap.data import build_dataset
from videoswap.models import build_model
from videoswap.pipelines import build_pipeline
from videoswap.utils.logger import MessageLogger, dict2str, reduce_loss_dict, set_path_logger
from videoswap.utils.vis_util import save_video_to_dir


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

    # Load the model components
    tokenizer = AutoTokenizer.from_pretrained(
        opt['path']['pretrained_model_path'],
        subfolder='tokenizer',
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        opt['path']['pretrained_model_path'],
        subfolder='text_encoder',
    )

    vae = AutoencoderKL.from_pretrained(
        opt['path']['pretrained_model_path'],
        subfolder='vae',
    )

    unet_type = opt['models']['unet'].pop('type')
    if unet_type == 'AnimateDiffUNet3DModel':
        inference_config_path = opt['models']['unet'].pop('inference_config_path')
        motion_module_path = opt['models']['unet'].pop('motion_module_path')
        unet = build_model(unet_type).from_pretrained_2d(
            opt['path']['pretrained_model_path'],
            subfolder='unet',
            unet_additional_kwargs=OmegaConf.to_container(OmegaConf.load(inference_config_path).unet_additional_kwargs),
        )
        motion_module_state_dict = torch.load(motion_module_path, map_location='cpu')
        motion_module_state_dict = {k.replace('.pos_encoder','.processor.pos_encoder'):v for k, v in motion_module_state_dict.items()}
        missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
    else:
        raise NotImplementedError

    adapter_type = opt['models']['adapter'].pop('type')
    t2i_adapter = build_model(adapter_type)(**OmegaConf.to_container(OmegaConf.load(opt['models']['adapter']['model_config_path'])))

    if opt.get('gradient_checkpointing'):
        print('enable gradient checkpointing in the training and testing')
        unet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # set up validation pipeline

    val_pipeline = build_pipeline(opt['val']['val_pipeline'])(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        adapter=t2i_adapter,
        scheduler=DDIMScheduler.from_pretrained(
            opt['path']['pretrained_model_path'],
            subfolder='scheduler',
        ))

    val_pipeline.enable_vae_slicing()
    val_pipeline.scheduler.set_timesteps(opt['val']['editing_config']['num_inference_steps'])

    # ----------------------------------------- set optimizer -----------------------------------------
    optim_opt = opt['train']['optimizer']
    optim_type = optim_opt.pop('type')
    assert optim_type == 'AdamW'

    optimizer = torch.optim.AdamW(t2i_adapter.parameters(), **optim_opt)

    # Prepare learning rate scheduler in accelerate config
    lr_scheduler = get_scheduler(
        opt['train']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=opt['train']['warmup_iter'],
        num_training_steps=opt['train']['total_iter'],
    )
    # ------------------------------------------------------------------

    # set up data loader (keep original and modify later)
    dataset_opt = opt['datasets']
    dataset_type = dataset_opt.pop('type')
    train_dataset = build_dataset(dataset_type)(dataset_opt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataset_opt['batch_size_per_gpu'],
        shuffle=True,
        num_workers=1,
    )
    # ---------------------------------------

    unet, t2i_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, t2i_adapter, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
        print('enable float16 in the training and testing')
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Start of config trainer
    train_pipeline = build_pipeline(opt['train']['train_pipeline'])(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        adapter=t2i_adapter,
        scheduler=DDPMScheduler.from_pretrained(
            opt['path']['pretrained_model_path'],
            subfolder='scheduler',
        ),
        # training hyperparams
        weight_dtype=weight_dtype,
        accelerator=accelerator,
        optimizer=optimizer,
        max_grad_norm=1.0,
        lr_scheduler=lr_scheduler,
        tune_cfg=opt['train'].get('tune_cfg', None)
    )
    train_pipeline.enable_vae_slicing()

    # Train!
    total_batch_size = opt['datasets']['batch_size_per_gpu'] * accelerator.num_processes
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}')
    logger.info(f"  Instantaneous batch size per device = {opt['datasets']['batch_size_per_gpu']}")
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f"  Total optimization steps = {opt['train']['total_iter']}")

    global_step = 0
    msg_logger = MessageLogger(opt, global_step)

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    # validation(unet, t2i_adapter, train_dataset, val_pipeline, opt, weight_dtype, global_step=0)

    while global_step < opt['train']['total_iter']:
        loss_dict = {}
        batch = next(train_data_yielder)

        """************************* start of an iteration*******************************"""

        loss = train_pipeline.step(batch)
        loss_dict['loss'] = loss

        log_dict = reduce_loss_dict(accelerator, loss_dict)

        # torch.cuda.empty_cache()
        """************************* end of an iteration*******************************"""
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step += 1

            if accelerator.is_main_process:
                if global_step % opt['logger']['print_freq'] == 0:
                    log_vars = {'iter': global_step}
                    log_vars.update({'lrs': lr_scheduler.get_last_lr()})
                    log_vars.update(log_dict)
                    msg_logger(log_vars)

                if global_step % opt['val']['val_freq'] == 0:
                    validation(unet, t2i_adapter, train_dataset, val_pipeline, opt, weight_dtype, global_step=global_step)

                if global_step % opt['logger']['save_checkpoint_freq'] == 0:
                    checkpoint_save_path = os.path.join(opt['path']['models'], f'models_{global_step}')
                    os.makedirs(checkpoint_save_path, exist_ok=True)
                    accelerator.save(t2i_adapter.state_dict(), os.path.join(checkpoint_save_path, 'adapter.pth'))
                    logger.info(f'save to {checkpoint_save_path}')


def validation(unet, t2i_adapter, train_dataset, val_pipeline, opt, weight_dtype, global_step=0):
    unet.eval()

    if t2i_adapter is not None and global_step != 0:
        t2i_adapter.eval()
        source_conditions = train_dataset.get_conditions()
    else:
        source_conditions = None

    # 2. load data
    source_frames = train_dataset.get_frames()
    edited_results = val_pipeline.validation(
        source_video=source_frames,
        source_conditions=source_conditions,
        source_prompt=opt['datasets']['prompt'],
        editing_config=opt['val']['editing_config'],
        dtype=weight_dtype,
        train_dataset=train_dataset,
        save_dir=opt['path']['visualization'])

    save_dir = os.path.join(opt['path']['visualization'], f'Iter_{global_step}', 'source')
    save_video_to_dir(source_frames, save_dir=save_dir, save_suffix='source', save_type=opt['val'].get('save_type', 'frame_gif'), fps=opt['val'].get('fps', 8))
    for key, edit_video in edited_results.items():
        if 'frame' not in opt['val'].get('save_type', 'frame_gif'):
            save_dir = os.path.join(opt['path']['visualization'], f'Iter_{global_step}')
        else:
            save_dir = os.path.join(opt['path']['visualization'], f'Iter_{global_step}', key)

        save_video_to_dir(edit_video, save_dir=save_dir, save_suffix=f"{key}_{opt['name']}", save_type=opt['val'].get('save_type', 'frame_gif'), fps=opt['val'].get('fps', 8))

    unet.train()
    if t2i_adapter is not None:
        t2i_adapter.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train_jeep.yml')
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train(root_path, args)
