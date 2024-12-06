import random

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, T2IAdapter
from diffusers.schedulers import KarrasDiffusionSchedulers
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from videoswap.models.animatediff_model import AnimateDiffUNet3DModel
from videoswap.pipelines.pipeline_videoswap import VideoSwapPipeline
from videoswap.utils.registry import PIPELINE_REGISTRY


def generate_sampleT(T_boundary, largeT_prob=1.0):
    if random.random() <= largeT_prob:  # < 0.7
        sample = random.uniform(T_boundary, 1)
    else:
        sample = random.uniform(0, T_boundary)
    return sample


@PIPELINE_REGISTRY.register()
class VideoSwapTrainer(VideoSwapPipeline):
    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer, unet: AnimateDiffUNet3DModel,
                 scheduler: KarrasDiffusionSchedulers, adapter: T2IAdapter, **kwargs):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, adapter)

        for name, module in kwargs.items():
            setattr(self, name, module)

    def step(self, batch: dict = dict()):
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()

        # Convert images to latent space
        images = batch['images'].to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, 'b c f h w -> (b f) c h w')
        latents = self.vae.encode(images).latent_dist.sample()
        latents = rearrange(latents, '(b f) c h w -> b c f h w', b=b)
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image

        timesteps = [int(generate_sampleT(self.tune_cfg['min_timestep']) * self.scheduler.config.num_train_timesteps) for _ in range(bsz)]
        timesteps = torch.tensor(timesteps).to(latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # get text ids
        text_input_ids = self.tokenizer(
            batch['prompt'],
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids.to(latents.device)
        encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        # condition
        adapter_state, loss_mask = self.adapter(batch['pred_tracks'], batch['img_size'], point_embedding=batch['point_embedding'], drop_rate=self.tune_cfg['drop_rate'], loss_type=self.tune_cfg['loss_type'])
        loss_mask = rearrange(loss_mask.unsqueeze(0), 'b f c h w -> b c f h w').to(batch['pred_tracks'].device)

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=adapter_state).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.scheduler.config.prediction_type}'
            )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        loss = ((loss * loss_mask).sum([1, 2, 3, 4]) / loss_mask.sum([1, 2, 3, 4])).mean()

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss
