import math
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange

if is_xformers_available():
    import xformers


class EDLoRA_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states
        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def revise_edlora_unet_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                layer.set_processor(EDLoRA_AttnProcessor(count))
                count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of edlora attention layer registered {cross_attention_idx}')


def bind_concept_prompt(prompts, new_concept_cfg):
    if isinstance(prompts, str):
        prompts = [prompts]
    new_prompts = []
    for prompt in prompts:
        prompt = [prompt] * 16
        for concept_name, new_token_cfg in new_concept_cfg.items():
            prompt = [
                p.replace(concept_name, new_name) for p, new_name in zip(prompt, new_token_cfg['concept_token_names'])
            ]
        new_prompts.extend(prompt)
    return new_prompts


def encode_edlora_prompt(
    pipe,
    prompt,
    new_concept_cfg,
    device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
):

    assert num_images_per_prompt == 1, 'only support num_images_per_prompt=1 now'

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:

        prompt_extend = bind_concept_prompt(prompt, new_concept_cfg)

        text_inputs = pipe.tokenizer(
            prompt_extend,
            padding='max_length',
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = pipe.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = rearrange(prompt_embeds, '(b n) m c -> b n m c', b=batch_size)

    prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)

    bs_embed, layer_num, seq_len, _ = prompt_embeds.shape

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !='
                            f' {type(prompt)}.')
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                ' the batch size of `prompt`.')
        else:
            uncond_tokens = negative_prompt

        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding='max_length',
            max_length=seq_len,
            truncation=True,
            return_tensors='pt',
        )

        negative_prompt_embeds = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)
        negative_prompt_embeds = (negative_prompt_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds


class LoRALinearLayer(nn.Module):
    def __init__(self, name, original_module, rank=4, alpha=1):
        super().__init__()

        self.name = name

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.lora_down = torch.nn.Conv2d(in_channels, rank, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(rank, out_channels, (1, 1), bias=False)
        else:
            in_features, out_features = original_module.in_features, original_module.out_features
            self.lora_down = nn.Linear(in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_features, bias=False)

        self.register_buffer('alpha', torch.tensor(alpha))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.original_forward = original_module.forward
        original_module.forward = self.forward

        self.enable_drop = False

    def forward(self, hidden_states):
        hidden_states = self.original_forward(hidden_states) + self.alpha * self.lora_up(self.lora_down(hidden_states))
        return hidden_states
