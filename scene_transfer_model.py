from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from typing import Any, Callable, List, Optional, Union
import PIL
import numpy as np
import torch

from scene_transfer.config import RunConfig
from scene_transfer import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX, attention_utils
from scene_transfer.sd15_transfer import SemanticAttentionSD15
from scene_transfer.model_utils import get_scene_transfer_sd15
from utils.adain import masked_adain, adain

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

class SceneTransfer:
    def __init__(self, config: RunConfig, pipe: Optional[SemanticAttentionSD15] = None):
        self.config = config
        self.pipe = get_scene_transfer_sd15() if pipe is None else pipe
        self.register_attention_control()
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.multi_swap_masks_32, self.multi_swap_masks_64 = None, None
        self.enable_edit = False
        self.step = 0

    def set_latents(self, latents_app: torch.Tensor, latents_struct: torch.Tensor):
        self.latents_app = latents_app
        self.latents_struct = latents_struct

    def set_noise(self, zs_app: torch.Tensor, zs_struct: torch.Tensor):
        self.zs_app = zs_app
        self.zs_struct = zs_struct

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_app_mask_32, self.image_struct_mask_32, self.image_app_mask_64, self.image_struct_mask_64 = masks

    def set_multi_swap_masks(self, matched_labels):
        self.multi_swap_masks_32, self.multi_swap_masks_64 = [], []
        
        # mark the pixels in target without semantic correspondences
        self.unmask_32 = torch.ones((32, 32), dtype=torch.bool, device=self.pipe.device)
        self.unmask_64 = torch.ones((64, 64), dtype=torch.bool, device=self.pipe.device)
        
        for label, source_mask, target_mask in matched_labels:
            
            source_mask_32 = F.interpolate(source_mask.unsqueeze(0).unsqueeze(0).float(), size=(32, 32), mode='nearest').squeeze(0).squeeze(0).bool()
            target_mask_32 = F.interpolate(target_mask.unsqueeze(0).unsqueeze(0).float(), size=(32, 32), mode='nearest').squeeze(0).squeeze(0).bool()
            source_mask_64 = F.interpolate(source_mask.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='nearest').squeeze(0).squeeze(0).bool()
            target_mask_64 = F.interpolate(target_mask.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='nearest').squeeze(0).squeeze(0).bool()
            
            # Update unmasked regions
            self.unmask_32 &= ~(target_mask_32)
            self.unmask_64 &= ~(target_mask_64)
            
            # Add multi_swap_masks_32 and multi_swap_masks_64 as model attributes if they don't exist
            if not hasattr(self.pipe, 'multi_swap_masks_32'):
                self.pipe.multi_swap_masks_32 = []
            if not hasattr(self.pipe, 'multi_swap_masks_64'):
                self.pipe.multi_swap_masks_64 = []
            
            # Append the resized masks to the model's attributes
            self.multi_swap_masks_32.append((label, source_mask_32, target_mask_32))
            self.multi_swap_masks_64.append((label, source_mask_64, target_mask_64))
        
            
    def prepare_attn_flow(self):
        self.attn_mask_32, self.attn_mask_64 = torch.zeros((32,32,32,32), device=self.pipe.device).bool(), torch.zeros((64,64,64,64), device=self.pipe.device).bool()
        for label, src_mask, tgt_mask in self.multi_swap_masks_32:
            src_indices = src_mask.nonzero(as_tuple=True)
            tgt_indices = tgt_mask.nonzero(as_tuple=True)
            for tgt_idx in zip(*tgt_indices):
                self.attn_mask_32[tgt_idx][src_indices] = True
            
        for label, src_mask, tgt_mask in self.multi_swap_masks_64:
            src_indices = src_mask.nonzero(as_tuple=True)
            tgt_indices = tgt_mask.nonzero(as_tuple=True)
            for tgt_idx in zip(*tgt_indices):
                self.attn_mask_64[tgt_idx][src_indices] = True
                
        self.attn_mask_32 = self.attn_mask_32.reshape(32**2, 32**2).float()
        self.attn_mask_64 = self.attn_mask_64.reshape(64**2, 64**2).float()
        
        self.attn_mask_32[self.unmask_32.flatten()] = True
        self.attn_mask_64[self.unmask_64.flatten()] = True
        
    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    latents[0] = masked_adain(latents[OUT_INDEX], latents[STYLE_INDEX], self.image_struct_mask_64, self.image_app_mask_64)
                else:
                    latents[0] = adain(latents[OUT_INDEX], latents[STYLE_INDEX])
            
        return callback

    def register_attention_control(self):

        model_self = self

        class AttentionProcessor:

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False,
                         t: int = 0,
                         vis: bool = False
                         ):

                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False

                # Potentially apply our cross image attention operation
                # To do so, we need to be in a self-attention layer in the decoder part of the denoising network
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True
                        if model_self.step % 5 == 0 and model_self.step < 40:
                            # Inject the structure's keys and values
                            key[OUT_INDEX] = key[STRUCT_INDEX]
                            value[OUT_INDEX] = value[STRUCT_INDEX]
                        else:
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX]
                            value[OUT_INDEX] = value[STYLE_INDEX]
                            
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                        query, key, value,
                        edit_map=perform_swap and model_self.enable_edit and should_mix,
                        is_cross=is_cross,
                        contrast_strength=model_self.config.contrast_strength,
                        )
                
                # Compute the cross attention and apply our contrasting operation
                if should_mix:
                    hidden_states_mix, attn_weight_mix = attention_utils.compute_scaled_dot_product_attention(
                        query, key, value,
                        edit_map=perform_swap and model_self.enable_edit and should_mix,
                        is_cross=is_cross,
                        contrast_strength=model_self.config.contrast_strength,
                        masks=[model_self.attn_mask_64, model_self.attn_mask_32] if hasattr(model_self, "attn_mask_64") else None
                        )
                    
                    hidden_states = hidden_states_mix
                
                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

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

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")