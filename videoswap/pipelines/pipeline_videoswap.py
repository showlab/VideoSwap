import copy
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, T2IAdapter
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import _preprocess_adapter_image
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging, randn_tensor
from einops import rearrange
from packaging import version
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer

from videoswap.models.animatediff_model import AnimateDiffUNet3DModel
from videoswap.utils.convert_edlora_to_diffusers import convert_edlora
from videoswap.utils.edlora_util import encode_edlora_prompt, revise_edlora_unet_attention_forward
from videoswap.utils.p2p_utils.attention_register import register_attention_control
from videoswap.utils.p2p_utils.attention_store import AttentionStore, EmptyControl
from videoswap.utils.p2p_utils.attention_util import make_controller
from videoswap.utils.p2p_utils.visualization import show_cross_attention
from videoswap.utils.registry import PIPELINE_REGISTRY

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray, List[Image.Image]]


@dataclass
class TuneAVideoInversionPipelineOutput(BaseOutput):
    latents: torch.FloatTensor


def visualize(frame_list, TAP_dict):

    pred_tracks = TAP_dict['pred_tracks']
    if 'index_list' in TAP_dict:
        index_list = TAP_dict['index_list']
    else:
        index_list = None

    res_frame_list = []

    point_nums = pred_tracks.shape[1]

    for idx, image in enumerate(frame_list):
        if idx >= len(pred_tracks):
            continue
        all_points = []
        all_colors = []

        pred_track_in_frame = pred_tracks[idx]

        for point_idx in range(point_nums):
            if index_list is not None and point_idx not in index_list:
                continue
            else:
                x, y = pred_track_in_frame[point_idx]
                if x >= 0 and y >= 0:
                    all_points.append(pred_track_in_frame[point_idx])
                    all_colors.append((0, 255, 0))
                else:
                    continue

        # 在图像上标记点
        draw = ImageDraw.Draw(image)
        radius = 5  # 圆点的半径
        for point, color in zip(all_points, all_colors):
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        res_frame_list.append(image)
    return res_frame_list


@PIPELINE_REGISTRY.register()
class VideoSwapPipeline(StableDiffusionPipeline):

    _optional_components = [
        'inverse_scheduler',
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: AnimateDiffUNet3DModel,
        scheduler: KarrasDiffusionSchedulers,
        adapter: T2IAdapter,
        inverse_scheduler: DDIMInverseScheduler = None
    ):

        if hasattr(scheduler.config,
                   'steps_offset') and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`'
                f' should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure '
                'to update the config accordingly as leaving `steps_offset` might led to incorrect results'
                ' in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,'
                ' it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`'
                ' file')
            deprecate('steps_offset!=1',
                      '1.0.0',
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config,
                   'clip_sample') and scheduler.config.clip_sample is True:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`.'
                ' `clip_sample` should be set to False in the configuration file. Please make sure to update the'
                ' config accordingly as not setting `clip_sample` in the config might lead to incorrect results in'
                ' future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very'
                ' nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
            deprecate('clip_sample not set',
                      '1.0.0',
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['clip_sample'] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, '_diffusers_version') and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse('0.9.0.dev0')  # noqa
        is_unet_sample_size_less_64 = hasattr(
            unet.config, 'sample_size') and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                'The configuration file of the unet has set the default `sample_size` to smaller than'
                ' 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the'
                ' following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-'
                ' CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5'
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                ' configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`'
                ' in the config might lead to incorrect results in future versions. If you have downloaded this'
                ' checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for'
                ' the `unet/config.json` file')
            deprecate('sample_size<64',
                      '1.0.0',
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(unet.config)
            new_config['sample_size'] = 64
            unet._internal_dict = FrozenDict(new_config)

        inverse_scheduler = DDIMInverseScheduler.from_config(scheduler.config)

        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, inverse_scheduler=inverse_scheduler, adapter=adapter)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.new_concept_cfg = None

        self.store_controller = AttentionStore()
        self.empty_controller = EmptyControl()

    def set_new_concept_cfg(self, new_concept_cfg=None):
        self.new_concept_cfg = new_concept_cfg
        self.tokenizer.new_concept_cfg = new_concept_cfg

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        video_length,
                        height,
                        width,
                        dtype,
                        device,
                        generator,
                        latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch'
                f' size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f'You have passed a list of generators of length {len(generator)}, but requested an effective batch'
                    f' size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i:i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        latents = rearrange(latents, '(b f) c h w -> b c f h w', b=1)

        return latents

    @torch.no_grad()
    def prepare_ddim_inverted_latents(self, video: List[Image.Image], prompt, num_inference_steps=50, LOW_RESOURCE=True, use_blend=False, dtype=torch.float16):
        # store attention, latent, feature when inversion
        if use_blend:
            register_attention_control(self, self.store_controller)
            resource_default_value = self.store_controller.LOW_RESOURCE
            self.store_controller.LOW_RESOURCE = LOW_RESOURCE
        else:
            self.store_controller = None

        ddim_latents = self.invert(prompt=prompt, video=video, num_inference_steps=num_inference_steps, controller=self.store_controller).latents

        if use_blend:
            # 3. remove controller
            register_attention_control(self, self.empty_controller)
            self.store_controller.LOW_RESOURCE = resource_default_value

        return ddim_latents

    def get_edit_controller(self, source_prompt, target_prompt, num_inference_steps, blend_words, blend_cfg, image_height, image_width):
        edit_controller = make_controller(
            tokenizer=self.tokenizer,
            prompts=[source_prompt, target_prompt],
            NUM_DDIM_STEPS=num_inference_steps,
            is_replace_controller=False,
            cross_replace_steps=blend_cfg.get('cross_replace_steps', 0.0),
            self_replace_steps=blend_cfg.get('self_replace_steps', 0.0),
            blend_words=blend_words,
            additional_attention_store=self.store_controller,
            blend_th=(blend_cfg.get('blend_th', 0.3), blend_cfg.get('blend_th', 0.3)),
            blend_self_attention=True,
            blend_latents=True,
            image_height=image_height,
            image_width=image_width)
        return edit_controller

    # input: video List[PIL.Image.Image]
    @torch.no_grad()
    def validation(self,
                   source_video,
                   source_conditions,
                   source_prompt,
                   editing_config,
                   dtype=torch.float16,
                   train_dataset=None,
                   save_dir=None
                   ):

        # 3. inversion (noise, store attention map, latent, features ...)
        use_invertion_latents = editing_config['use_invertion_latents']
        use_blend = editing_config.get('use_blend', False)
        visualize_point = editing_config.get('visualize_point', False)
        visualize_attention = editing_config.get('visualize_attention', False)

        if use_invertion_latents:
            ddim_latents = self.prepare_ddim_inverted_latents(
                video=source_video,
                prompt=source_prompt,
                num_inference_steps=editing_config['num_inference_steps'],
                LOW_RESOURCE=True,  # not classifier-free guidance
                use_blend=use_blend,
                dtype=dtype
            )
        else:
            ddim_latents = None

        ddim_latents = ddim_latents.to(dtype=dtype)

        pretrained_unet_state_dict = copy.deepcopy(self.unet.state_dict())
        pretrained_text_encoder_dict = copy.deepcopy(self.text_encoder.state_dict())
        del pretrained_text_encoder_dict['text_model.embeddings.token_embedding.weight']

        # 4. perform_editing
        edited_results = {}

        for key, swap_cfg in editing_config['editing_prompts'].items():

            lora_path = swap_cfg.get('lora_path', None)
            if lora_path is not None:
                lora_path, lora_alpha = lora_path.split('---')
                enable_edlora = 'edlora' in lora_path
                _, new_concept_cfg = convert_edlora(self, torch.load(lora_path), enable_edlora=enable_edlora, alpha=float(lora_alpha))
                if enable_edlora:
                    logger.info(f'loading edlora: {lora_path}, using alpha={lora_alpha}')
                    revise_edlora_unet_attention_forward(self.unet)
                    self.set_new_concept_cfg(new_concept_cfg)

            if source_conditions is not None and swap_cfg.get('tap_path'):
                conditions = train_dataset.get_conditions(swap_cfg['tap_path'])
            else:
                conditions = copy.deepcopy(source_conditions)

            if conditions is not None and swap_cfg.get('select_point'):
                index_list = []
                for select_point_name in swap_cfg['select_point']:
                    index_list.append(conditions['point_name2id'][select_point_name])
                conditions['index_list'] = index_list
            else:
                if conditions is not None:
                    conditions['index_list'] = None

            # ---------------------------------------------------------------
            source_subject, target_subject = swap_cfg['replace'].split('->')
            source_subject, target_subject = source_subject.strip(), target_subject.strip()
            assert source_subject in source_prompt, 'source subject need in source prompt'
            target_prompt = source_prompt.replace(source_subject, target_subject)

            if 'replace_other' in swap_cfg:
                source_other, target_other = swap_cfg['replace_other'].split('->')
                source_other, target_other = source_other.strip(), target_other.strip()
                assert source_other in target_prompt, 'source subject need in source prompt'
                target_prompt = target_prompt.replace(source_other, target_other)

            if use_blend:
                width, height = source_video[0].size
                blend_words = [source_subject.split(' '), target_subject.split(' ')]

                blend_cfg = swap_cfg.get('blend_cfg', {})
                edit_controller = self.get_edit_controller(source_prompt, target_prompt, editing_config['num_inference_steps'], blend_words=blend_words, blend_cfg=blend_cfg, image_height=height, image_width=width)
                register_attention_control(self, edit_controller)
            elif visualize_attention:
                edit_controller = AttentionStore()
                register_attention_control(self, edit_controller)
            else:
                edit_controller = None

            if 't2i_guidance_scale' in swap_cfg:
                t2i_guidance_scale = swap_cfg['t2i_guidance_scale']
            else:
                t2i_guidance_scale = editing_config.get('t2i_guidance_scale', 1.0)

            if 'guidance_scale' in swap_cfg:
                guidance_scale = swap_cfg['guidance_scale']
            else:
                guidance_scale = editing_config.get('guidance_scale', 7.5)

            if 'negative_prompt' in swap_cfg:
                negative_prompt = swap_cfg['negative_prompt']
            else:
                negative_prompt = editing_config.get('negative_prompt', None)

            sequence_return = self(
                prompt=target_prompt,
                conditions=conditions,
                source_prompt=source_prompt,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device='cuda').manual_seed(0),
                num_inference_steps=editing_config['num_inference_steps'],
                video_length=len(source_video),
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                # used in null inversion
                latents=ddim_latents,
                uncond_embeddings_list=None,
                controller=edit_controller,
                # t2i_cfg
                t2i_guidance_scale=t2i_guidance_scale,
                t2i_start=editing_config.get('t2i_start', 0.0),
                t2i_end=editing_config.get('t2i_end', 1.0),
            )

            sequence = sequence_return.videos

            edited_results[key] = copy.deepcopy(sequence)

            if conditions is not None and visualize_point:
                sequence = visualize(sequence, conditions)
                edited_results[key + '_vispoint'] = sequence

            if visualize_attention and len(edit_controller.attention_store.keys()) > 0:
                assert save_dir is not None
                target_subject_token = ' '.join([f'<{concept_token}_0>' for concept_token in target_subject.split() if concept_token.startswith('<') and concept_token.endswith('>')])

                attention_save_dir = os.path.join(save_dir, f'{key}_attention')
                os.makedirs(attention_save_dir, exist_ok=True)
                _ = show_cross_attention(
                    self.tokenizer,
                    target_prompt.replace(target_subject, target_subject_token),
                    edit_controller,
                    res_y=14, res_x=24, from_where=['up', 'down'],
                    save_dir=attention_save_dir)

            if lora_path is not None:
                self.unet.load_state_dict(pretrained_unet_state_dict)
                self.text_encoder.load_state_dict(pretrained_text_encoder_dict, strict=False)
                self.set_new_concept_cfg(None)
                logger.info('remove unet lora and text encoder lora')

        return edited_results

    # latents+text sample, text-only sample, output: video List[PIL.Image.Image]
    @torch.no_grad()
    def __call__(self,
                 prompt: Union[str, List[str]],
                 conditions,
                 video_length: Optional[int],
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = 'pil',
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: int = 1,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0.0,
                 controller=None,
                 t2i_guidance_scale=1.0,
                 t2i_start=0.0,
                 t2i_end=1.0,
                 **args):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (cross_attention_kwargs.get('scale', None) if cross_attention_kwargs is not None else None)

        if self.new_concept_cfg is not None:
            prompt_embeds = encode_edlora_prompt(
                self,
                prompt,
                self.new_concept_cfg,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
        else:
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if conditions is not None:
            if isinstance(conditions, dict):
                if 'point_embedding' in conditions:
                    point_embedding = conditions['point_embedding'].to(self.device, dtype=latents_dtype)
                else:
                    point_embedding = None

                adapter_state = self.adapter(
                    conditions['pred_tracks'].to(self.device, dtype=latents_dtype),
                    conditions['img_size'],
                    point_embedding=point_embedding,
                    index_list=conditions['index_list']
                )
            else:
                width, height = conditions[0].size

                condition_input = _preprocess_adapter_image(conditions, height, width).to(self.device, dtype=latents_dtype)
                adapter_state = self.adapter(condition_input)

            for k, v in enumerate(adapter_state):
                adapter_state[k] = v * t2i_guidance_scale
            if num_images_per_prompt > 1:
                raise NotImplementedError
            if do_classifier_free_guidance:
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = torch.cat([v] * 2, dim=0)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if conditions is not None and i <= len(timesteps) * t2i_end and i >= len(timesteps) * t2i_start:
                    t2i_residual = [state.clone() for state in adapter_state]
                else:
                    t2i_residual = None

                # predict the noise residual

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=t2i_residual,
                    return_dict=False,
                )[0].to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Edit the latents using attention map
                if controller is not None:
                    dtype = latents.dtype
                    latents_new = controller.step_callback(latents)
                    latents = latents_new.to(dtype)

                    # print(latents.shape) # torch.Size([1, 4, 16, 56, 96])

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = rearrange(latents, 'b c f h w -> (b f) c h w')

        if not output_type == 'latent':
            video = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            video = latents

        video = self.image_processor.postprocess(video, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video

        return TuneAVideoPipelineOutput(videos=video)

    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[str] = None,
        video: List[PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_guidance_amount: float = 0.1,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controller=None
    ):
        # 1. Define call parameters

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess image
        video = self.image_processor.preprocess(video)

        # 4. Prepare latent variables
        latents = self.prepare_image_latents(video, video.shape[0], self.vae.dtype, device, generator)

        # 5. Encode input prompt
        num_images_per_prompt = 1
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        # self.unet = prepare_unet(self.unet)

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    # cross_attention_kwargs={"timestep": t},
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                # Edit the latents using attention map
                if controller is not None:
                    dtype = latents.dtype
                    latents_new = controller.step_callback(latents)
                    latents = latents_new.to(dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        inverted_latents = latents.detach().clone()

        # Offload last model to CPU
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (inverted_latents)

        return TuneAVideoInversionPipelineOutput(latents=inverted_latents)
