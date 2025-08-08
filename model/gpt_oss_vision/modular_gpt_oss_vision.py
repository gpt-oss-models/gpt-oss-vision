# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ...cache_utils import Cache, DynamicCache
from ...integrations.hub_kernels import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import OutputRecorder, check_model_inputs

# Llama / Mixtral / Qwen2 base building blocks (keeps parity with original GPT-OSS codebase).
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from ..qwen2.modeling_qwen2 import Qwen2Attention

# Use the Vision-enabled config (user provided).
try:
    # Prefer the vision-specific config if present in package
    from .configuration_gpt_oss_vision import GPTOSSVisionConfig as GptOssConfig
except Exception:
    # fallback to previous-named config if necessary
    from .configuration_gpt_oss import GptOssConfig

logger = logging.get_logger(__name__)

# -----------------------------------------------------------------------------
# Vision Adapter
# -----------------------------------------------------------------------------
class VisionAdapter(nn.Module):
    """
    Wraps a ViT encoder (or compatible) and a projection to model.hidden_size.
    Optionally provides a pooled prefix token or returns the full patch embeddings.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        # Try robust imports for ViT model
        try:
            # When used externally, 'transformers.models.vit' is usually available
            from transformers.models.vit.modeling_vit import ViTModel

            self.vit = ViTModel(config=type("VitCfg", (), {
                "image_size": getattr(config, "vision_image_size", 224),
                "hidden_size": config.vision_embed_dim,
                "patch_size": config.vision_patch_size,
                "num_channels": config.vision_num_channels,
                "num_hidden_layers": config.vision_num_layers,
                "num_attention_heads": config.vision_num_heads,
            })())
        except Exception:
            # Last resort: create a minimal conv patcher (very lightweight)
            # This fallback is deliberately simplistic — replace with proper ViT in repo.
            self.vit = None
            logger.warning("transformers ViTModel not available — VisionAdapter will use a simple conv patch embed fallback.")

            patch_size = config.vision_patch_size
            in_ch = config.vision_num_channels
            hidden = config.vision_embed_dim
            self.patch_embed = nn.Conv2d(in_ch, hidden, kernel_size=patch_size, stride=patch_size)

        # Projection from vision embed_dim -> model hidden_size
        self.proj = nn.Linear(config.vision_embed_dim, config.hidden_size)

        # Pool token or prefix: optional; add one learnable prefix token (can be extended)
        self.use_pool = getattr(config, "vision_pool_token", True)
        if self.use_pool:
            self.pool_token = nn.Parameter(torch.randn(1, 1, config.vision_embed_dim))
        else:
            self.pool_token = None

        # Optional small layernorm
        self.norm = nn.LayerNorm(config.vision_embed_dim)

        # initialization
        self._init_weights()

    def _init_weights(self):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(self.proj, nn.Linear):
            self.proj.weight.data.normal_(mean=0.0, std=std)
            if self.proj.bias is not None:
                self.proj.bias.data.zero_()
        if self.pool_token is not None:
            nn.init.normal_(self.pool_token, mean=0.0, std=std)

    def forward(self, pixel_values: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
            pixel_values: (batch, channels, height, width)
        Returns:
            vision_embeds: (batch, vision_seq_len, hidden_size) ready to concatenate with text embeddings.
        """
        if pixel_values is None:
            return None

        if self.vit is not None:
            outputs = self.vit(pixel_values)
            # ViTModel typically returns last_hidden_state (batch, seq_len, vit_hidden)
            vit_embeds = outputs.last_hidden_state  # includes class token at index 0 if configured
        else:
            # fallback: patchify via conv
            x = self.patch_embed(pixel_values)  # (batch, C_out, H', W')
            b, c, h, w = x.shape
            vit_embeds = x.flatten(2).transpose(1, 2)  # (batch, seq, c)

        # optionally pool / replace class token
        if self.pool_token is not None:
            # pool token expand and prepend
            pool = self.pool_token.expand(vit_embeds.shape[0], -1, -1)  # (batch, 1, vit_hidden)
            # For fallback conv, there's no class token to replace, so just prepend
            vit_embeds = torch.cat([pool, vit_embeds], dim=1)

        # normative projection to model hidden size
        vit_embeds = self.norm(vit_embeds)
        projected = self.proj(vit_embeds)  # (batch, seq, hidden_size)
        return projected


# -----------------------------------------------------------------------------
# Model building blocks — largely preserved from original modular file
# -----------------------------------------------------------------------------
class GptOssRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama


class GptOssExperts(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        Same algorithm as original: training loops over experts, inference repeats for efficiency.
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states


class GptOssTopKRouter(nn.Module):
    """
    Top-K router with optional vision bias applied to router logits for vision tokens.

    New:
    - vision_bias: learned per-expert bias applied only to vision tokens
    - forward accepts modality_mask (batch, seq_len) bool where True indicates vision tokens
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))
        self.vision_bias = nn.Parameter(torch.zeros(self.num_experts))  # small learned bias for vision tokens

    def forward(self, hidden_states: torch.Tensor, modality_mask: Optional[torch.BoolTensor] = None) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        hidden_states: (batch, seq_len, hidden_dim) or flattened (num_tokens, hidden_dim)
        modality_mask: (batch, seq_len) bool where True = vision token

        Returns:
            router_scores: (num_tokens, num_experts) dense soft allocations
            router_indices: (num_tokens, top_k) top-k chosen expert indices
        """
        flat = False
        if hidden_states.dim() == 3:
            batch, seq, dim = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            if modality_mask is not None:
                modality_mask = modality_mask.reshape(-1)
            flat = True

        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (num_tokens, num_experts)

        # Apply vision bias where modality_mask is True
        if modality_mask is not None:
            if modality_mask.dtype != torch.bool:
                modality_mask = modality_mask.to(torch.bool)
            # Expand vision_bias to tokens and add to logits for vision tokens only
            if modality_mask.dim() == 1:
                router_logits[modality_mask] = router_logits[modality_mask] + self.vision_bias.to(router_logits.dtype)
            else:
                # unexpected shape — flatten
                mm = modality_mask.reshape(-1).to(torch.bool)
                router_logits[mm] = router_logits[mm] + self.vision_bias.to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (num_tokens, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states: torch.Tensor, modality_mask: Optional[torch.BoolTensor] = None):
        # Router can use modality_mask to bias vision tokens towards certain experts
        router_scores, router_indices = self.router(hidden_states, modality_mask=modality_mask)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores


class GptOssRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # prevent overflow in BF16/FP16 for bsz>1
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GptOssAttention(Qwen2Attention):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # apply rotary
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config=config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, kept for BC
        modality_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # pass modality_mask to MLP/router to enable vision bias specialization
        hidden_states, _ = self.mlp(hidden_states, modality_mask=modality_mask)
        hidden_states = residual + hidden_states
        return hidden_states


class GptOssPreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
    _supports_sdpa = False
    _supports_flash_attention = False
    _supports_flex_attention = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(GptOssTopKRouter, index=0),
        "hidden_states": GptOssDecoderLayer,
        "attentions": GptOssAttention,
    }

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GptOssRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GptOssExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.gate_up_proj_bias.data.zero_()
            module.down_proj.data.normal_(mean=0.0, std=std)
            module.down_proj_bias.data.zero_()
        elif isinstance(module, GptOssAttention):
            module.sinks.data.normal_(mean=0.0, std=std)
        elif isinstance(module, GptOssTopKRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.bias.data.normal_(mean=0.0, std=std)
            if hasattr(module, "vision_bias"):
                module.vision_bias.data.zero_()


class GptOssModel(MixtralModel):
    _no_split_modules = ["GptOssDecoderLayer"]

    def __init__(self, config: GptOssConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        # Vision adapter created only if config.use_vision True
        if getattr(self.config, "use_vision", False):
            self.vision_adapter = VisionAdapter(self.config)
        else:
            self.vision_adapter = None

        # Rotary embedding specialized for GptOss
        self.rotary_emb = GptOssRotaryEmbedding(config=self.config)

        # Ensure weights init from parent is preserved
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,  # new multimodal input
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        # --- Input validation / embedding ---
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Get text token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute cache_position & defaults for position_ids
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # If vision enabled and pixel_values provided, produce vision embeddings and prepend them
        modality_mask = None
        if self.vision_adapter is not None and pixel_values is not None:
            vision_embeds = self.vision_adapter(pixel_values)  # (batch, vis_seq_len, hidden_size)
            if vision_embeds is None:
                raise ValueError("Vision adapter returned None. Ensure ViT is available or provide pixel_values correctly.")
            
            # Prepend vision embeds to inputs_embeds
            inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
            
            # Update attention mask: if not provided create ones
            batch_size = inputs_embeds.shape[0]
            vis_len = vision_embeds.shape[1]
            if attention_mask is None:
                # create ones for text tokens after vision tokens
                attention_mask = torch.ones(batch_size, inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)
                # but since we have full sequence, it's all ones
            else:
                # If attention_mask provided for text tokens only, prepend ones for vision tokens
                if attention_mask.dim() == 2 and attention_mask.shape[1] == (inputs_embeds.shape[1] - vis_len):
                    vis_ones = torch.ones(batch_size, vis_len, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([vis_ones, attention_mask], dim=1)
            
            # Update position_ids to account for vision tokens
            # Vision tokens should start from position 0, text tokens continue from there
            if position_ids is not None:
                # Create new position_ids that start from 0 for vision tokens
                new_position_ids = torch.arange(inputs_embeds.shape[1], device=position_ids.device, dtype=position_ids.dtype)
                new_position_ids = new_position_ids.unsqueeze(0).expand(batch_size, -1)
                position_ids = new_position_ids
            else:
                # If position_ids was None, create it for the full sequence
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # modality_mask: True==vision, False==text
            modality_mask = torch.zeros(batch_size, inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
            modality_mask[:, :vis_len] = True

        # Prepare causal masks mapping (full & sliding)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # Compute base rotary embeddings (cos, sin) using current positions
        # Note: we'll allow NoPE to override these per-layer if requested.
        base_cos, base_sin = self.rotary_emb(hidden_states, position_ids)

        # Iterate layers and apply NoPE logic per PRD
        for layer_idx, decoder_layer in enumerate(self.layers):
            # Determine whether this layer should use NoPE (neutralized RoPE)
            if getattr(self.config, "use_nope", False) and getattr(self.config, "nope_stride", None):
                stride = int(self.config.nope_stride)
                if stride > 0 and ((layer_idx + 1) % stride == 0):
                    # Neutralize RoPE: cos=1, sin=0 broadcasted to same shape and dtype
                    cos = torch.ones_like(base_cos)
                    sin = torch.zeros_like(base_sin)
                else:
                    cos, sin = base_cos, base_sin
            else:
                cos, sin = base_cos, base_sin

            position_embeddings = (cos, sin)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                modality_mask=modality_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class GptOssForCausalLM(MixtralForCausalLM):
    pass


__all__ = ["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel"]
