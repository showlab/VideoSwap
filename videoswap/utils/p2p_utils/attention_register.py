import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange

from videoswap.utils.edlora_util import EDLoRA_AttnProcessor

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class EDLoRA_AttnControlProcessor:
    def __init__(self, cross_attention_idx, place_in_unet, controller, attention_op=None):
        self.attention_op = attention_op
        self.place_in_unet = place_in_unet
        self.controller = controller
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

        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:                                      # single layer embedding
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

        if is_xformers_available() and query.shape[-2] >= 32**2:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            attention_probs = rearrange(attention_probs, '(b h) s t -> b h s t', h=attn.heads)
            attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
            attention_probs = rearrange(attention_probs, 'b h s t -> (b h) s t', h=attn.heads)

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


class AttnControlProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, place_in_unet, controller, attention_op=None):
        self.place_in_unet = place_in_unet
        self.controller = controller
        self.attention_op = attention_op

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

        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xformers_available() and query.shape[-2] >= 32**2:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            attention_probs = rearrange(attention_probs, '(b h) s t -> b h s t', h=attn.heads)
            attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
            attention_probs = rearrange(attention_probs, 'b h s t -> (b h) s t', h=attn.heads)

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


def register_attention_control(model, controller):
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def change_forward(unet, count_self, count_cross, place_in_unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and ('attn1' in name or 'attn2' in name):
                if isinstance(layer.processor, (AttnProcessor, AttnControlProcessor, XFormersAttnProcessor, AttnProcessor2_0)):
                    layer.set_processor(AttnControlProcessor(place_in_unet, controller))
                elif isinstance(layer.processor, (EDLoRA_AttnProcessor, EDLoRA_AttnControlProcessor)):
                    layer.set_processor(EDLoRA_AttnControlProcessor(count_cross, place_in_unet, controller))
                else:
                    print(layer.processor)
                    raise NotImplementedError

                if 'attn1' in name:
                    count_self += 1
                else:
                    count_cross += 1
            else:
                count_self, count_cross = change_forward(layer, count_self, count_cross, place_in_unet)
        return count_self, count_cross

    # use this to ensure the order
    count_self, count_cross = change_forward(model.unet.down_blocks, 0, 0, 'down')
    count_self, count_cross = change_forward(model.unet.mid_block, count_self, count_cross, 'mid')
    count_self, count_cross = change_forward(model.unet.up_blocks, count_self, count_cross, 'up')
    print(f'Number of attention control layer registered {count_cross + count_self}')
    controller.num_att_layers = count_cross + count_self
