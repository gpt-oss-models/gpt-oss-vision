# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Based on the original GPT-OSS configuration from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Renamed to GPTOSSVisionConfig for multimodal (text + image) support
# - Added vision encoder parameters
# - Added NoPE (No Positional Embedding in periodic layers) configuration
# - Updated default settings for multimodal pretraining
# - Contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""GPT-OSS-Vision model configuration (multimodal + NoPE support)"""

from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation


class GPTOSSVisionConfig(PretrainedConfig):
    r"""
    Configuration for the GPT-OSS-Vision model.

    Supports:
    - Text + Vision inputs
    - Optional NoPE (No Positional Embedding) in attention layers
    - MOE (Mixture of Experts) configuration from GPT-OSS base

    Args:
        num_hidden_layers (int, *optional*, defaults to 36):
            Number of Transformer layers.
        vocab_size (int, *optional*, defaults to 201088):
            Vocabulary size for text tokens.
        hidden_size (int, *optional*, defaults to 2880):
            Dimensionality of hidden states.
        intermediate_size (int, *optional*, defaults to 2880):
            Feed-forward layer size.
        vision_embed_dim (int, *optional*, defaults to 1024):
            Embedding size for image features from vision encoder.
        vision_patch_size (int, *optional*, defaults to 14):
            Patch size for vision transformer encoder.
        vision_num_channels (int, *optional*, defaults to 3):
            Number of channels in vision input (default RGB).
        use_nope (bool, *optional*, defaults to False):
            Whether to enable NoPE instead of RoPE in certain layers.
        nope_stride (int, *optional*, defaults to 4):
            Frequency (in layers) to skip positional embeddings when use_nope=True.
    """

    model_type = "gpt_oss_vision"
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "vision_embed": (["pixel_values"], ["vision_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
    }

    def __init__(
        self,
        # Text params
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},

        # Vision params
        vision_embed_dim: int = 1024,
        vision_patch_size: int = 14,
        vision_num_channels: int = 3,
        vision_num_layers: int = 24,
        vision_num_heads: int = 16,

        # NoPE params
        use_nope: bool = False,
        nope_stride: int = 4,

        # Common params
        tie_word_embeddings: bool = False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits: bool = False,
        use_cache: bool = True,
        layer_types=None,
        **kwargs,
    ):
        # ---- Text parameters ----
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_local_experts = num_local_experts
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        # ---- Vision parameters ----
        self.vision_embed_dim = vision_embed_dim
        self.vision_patch_size = vision_patch_size
        self.vision_num_channels = vision_num_channels
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads

        # ---- NoPE parameters ----
        self.use_nope = use_nope
        self.nope_stride = nope_stride

        # ---- Layer types ----
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        # ---- RoPE / NoPE validation ----
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # ---- Misc ----
        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["GPTOSSVisionConfig"]