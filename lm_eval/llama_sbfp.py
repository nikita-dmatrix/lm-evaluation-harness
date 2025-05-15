import numpy as np
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import scipy

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers import Cache
from qtorch import quant


def quantize_bfp(tensor, bits = 4, block_size = 128, along_rows=False):
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        if (shape[-1] % block_size) != 0:
            #print(shape)
            tensor = F.pad(tensor, (0, block_size - (shape[-1] % block_size )), "constant", 0)
        new_shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*new_shape)
        tensor = tensor[..., : shape[-1]]
        tensor = tensor.transpose(-1,-2)
    return tensor



low_cutoff = 1./128
high_cutoff = 496



def extract_4bit_rounded_mantissa_and_exp(x):
    x32  = x.float().view(dtype=torch.int32)
    exp = (x32 & 0x7F800000 ) >> 23
    man8 = (((x32 & 0x007C0000) >> 17) | ((x32 & 0x0003FFFF) != 0)).to(torch.int8)
    man8[torch.logical_or(((man8 & 0x3) == 0x3), ((man8 & 0x6) == 0x6))] += 0x4
    man8 >>= 2
    exp[(man8 & 0x10) != 0] += 1
    man8[(man8 & 0x10) != 0] = 0

    return man8, exp


def float_to_e4m4(x, ebias):
    man8, exp = extract_4bit_rounded_mantissa_and_exp(x)
    exp = exp - 127 + ebias
    man8[exp < 0] = 0
    exp[exp < 0] = 0

    man8[exp > 15] = 15
    exp[exp > 15] = 15

    if x.dtype == torch.float32:
        answ = ((man8.to(torch.int32) << 19) | ((exp + 127 - ebias).to(torch.int32) << 23)).view(dtype=torch.float32)
    elif x.dtype == torch.bfloat16:
        answ = ((man8.to(torch.int16) << 3) | ((exp + 127 - ebias).to(torch.int16) << 7)).view(dtype=torch.bfloat16)
    else:
        raise NotImplementedError
    return answ

def float_to_fp4(x):
    mask_done = torch.zeros_like(x, dtype=torch.bool)
    sign = x < 0
    x = torch.abs(x)
    mask_6 = x > 5.0
    mask_4 = x >= 3.5
    mask_3 = x > 2.5
    mask_2 = x >= 1.75
    mask_15 = x > 1.25
    mask_1 = x >= 0.75
    mask_05 = x > 0.25

    x[mask_6] = 6.0
    mask_done |= mask_6
    x[torch.logical_and(mask_4,  torch.logical_not(mask_done))] = 4.0
    mask_done |= mask_4
    x[torch.logical_and(mask_3,  torch.logical_not(mask_done))] = 3.0
    mask_done |= mask_3
    x[torch.logical_and(mask_2,  torch.logical_not(mask_done))] = 2.0
    mask_done |= mask_2
    x[torch.logical_and(mask_15,  torch.logical_not(mask_done))] = 1.5
    mask_done |= mask_15
    x[torch.logical_and(mask_1,  torch.logical_not(mask_done))] = 1.0
    mask_done |= mask_1
    x[torch.logical_and(mask_05,  torch.logical_not(mask_done))] = 0.5
    mask_done |= mask_05
    x[torch.logical_not(mask_done)] = 0.0
    x[sign] *= -1.0
    return x

def getSBFPEbias(x):
    x = torch.max(torch.abs(x))/7.0
    man8, exp = extract_4bit_rounded_mantissa_and_exp(x)
    exp -= 127
    exp = exp.detach().cpu().item()
    ebias = 15 - (exp if exp > 0 else 0)
    ebias = ebias if ebias > 1 else 1
    return ebias


def quantize_sort(tensor, bits = 4, block_size = 16, along_rows=True, ebias=None, use_sfp=False, wrap_bfp=True):
    if along_rows:
        norm = torch.norm(tensor, dim=0)
        order = torch.argsort(norm)
        #order = torch.arange(len(norm), device=norm.device)
        reverse = torch.empty_like(order)
        reverse[order] = torch.arange(len(order), device=order.device)
        tensor = tensor[:,order]
        tensor = quantize_sbfp(tensor, bits, block_size, along_rows, ebias, use_sfp)
        tensor = quantize_bfp(tensor, 8, 64, along_rows)
        tensor = tensor[:,reverse]
        return tensor
    else:
        raise NotImplementedError

def quantize_sbfp(tensor, bits = 4, block_size = 128, along_rows=False, ebias=None, use_sfp=False):
    if ebias is None:
        ebias = getSBFPEbias(tensor)
        print("ebias", ebias)

    maxrepr = (2 ** (bits - 1) - 1.)
    if use_sfp:
        assert bits == 4
        maxrepr = 6.0

    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        maxx = torch.max(torch.abs(tensor), dim = -1, keepdims=True).values
        scale = maxx / maxrepr

        #print("before", scale)
        scale = float_to_e4m4(scale, ebias)
        #print("after", scale)
        if use_sfp:
            tensor = float_to_fp4(tensor/scale)
        else:
            tensor = torch.round(tensor / scale)
            tensor[tensor > maxrepr] = maxrepr
            tensor[tensor < -maxrepr] = -maxrepr

        #print ("tensormax", tensor.max())
        #print(tensor)
        tensor *= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape

        #assert shape[-1] % block_size == 0, f"shape {shape} not divisible by block_size"
        if (shape[-1] % block_size) != 0:
            #print(shape)
            tensor = F.pad(tensor, (0, block_size - (shape[-1] % block_size )), "constant", 0)
        new_shape = tensor.shape

        tensor = tensor.reshape(-1, block_size)


        maxx = torch.max(torch.abs(tensor), dim = -1, keepdims=True).values
        #maxx [ maxx < 6e-8] = 1.
        scale = maxx / maxrepr
        scale = float_to_e4m4(scale, ebias)
        #print(tensor)
        #print(scale.shape)

        #print(scale.min(), scale.max())
        if use_sfp:
            tensor = float_to_fp4(tensor/scale)
        else:
            tensor = torch.round(tensor / scale)
            tensor[tensor > maxrepr] = maxrepr
            tensor[tensor < -maxrepr] = -maxrepr

        #print(tensor)

        tensor *= scale

        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = tensor.reshape(*new_shape)
        #print(tensor.shape)
        tensor = tensor[..., : shape[-1]]
        #print(tensor.shape)
        #print("=====")
        return tensor.transpose(-1,-2)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionWrapperQK(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attention, quantize=False, reorder=False, pnorm=2, block_size=128):
        super().__init__()
        for attr, val in attention.__dict__.items():
            #print(attr)
            self.__setattr__(attr, val)
        #self.Kactivation = None

        self.quantize = quantize
        self.reorder = reorder
        self.block_size=block_size

        self.dtype = self.k_proj.get_parameter('weight').data.dtype




    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #self.Kactivation = key_states.cpu()

        if self.quantize:
            #UNUSED WRAPPER!
            #query_states = quantize_bfp(query_states.float(), 8, self.block_size, True).to(self.dtype)
            key_states = quantize_sbfp(key_states.float(), 4, self.block_size, True).to(self.dtype)
            value_states = quantize_sbfp(value_states.float(), 4, self.block_size, False).to(self.dtype)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
class LlamaSDPAAttentionWrapper(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attention, quantize=False, reorder=False, pnorm=2, block_size=128):
        super().__init__()
        for attr, val in attention.__dict__.items():
            #print(attr)
            self.__setattr__(attr, val)
        #self.Kactivation = None

        self.quantize = quantize
        self.reorder = reorder
        self.block_size=block_size

        self.dtype = self.k_proj.get_parameter('weight').data.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
       query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.quantize:
            #query_states = quantize_bfp(query_states.float(), 8, self.block_size, True).to(self.dtype)
            #key_states = quantize_sbfp(key_states.float(), 4, self.block_size, True).to(self.dtype)
            #value_states = quantize_sbfp(value_states.float(), 4, self.block_size, False).to(self.dtype)
            #key_states = quantize_sbfp(key_states.float(), 4, self.block_size, True)
            key_states = quantize_bfp(key_states.float(),8,64,True).to(self.dtype)
            value_states = quantize_sbfp(value_states.float(), 4, self.block_size, False)
            value_states = quantize_bfp(value_states.float(),8,64,False).to(self.dtype)
            query_states = quantize_bfp(query_states.float(), 8, 64, True).to(self.dtype)


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

def wrap_model(model, quantize = False, reorder=False, pnorm=2, block_size=128):
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].self_attn = LlamaSDPAAttentionWrapper(model.model.layers[idx].self_attn, quantize, reorder, pnorm, block_size)


def quantize_weight(W):
    #return quantize_sbfp(W.float(), 4, 16, True).to(W.dtype).to(W.device)
    #return quantize_bfp(W.float(), 8, 64, True).to(W.dtype).to(W.device)
    return quantize_sort(W.float(), 4, 16, True, use_sfp=False).to(W.dtype).to(W.device)



def wrap_model_weights(model):
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].self_attn.q_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.q_proj.weight.data)
        model.model.layers[idx].self_attn.k_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.k_proj.weight.data)
        model.model.layers[idx].self_attn.v_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.v_proj.weight.data)
        model.model.layers[idx].self_attn.o_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.o_proj.weight.data)
        model.model.layers[idx].mlp.gate_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.gate_proj.weight.data)
        model.model.layers[idx].mlp.up_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.up_proj.weight.data)
        model.model.layers[idx].mlp.down_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.down_proj.weight.data)




