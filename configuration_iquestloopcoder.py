# Copyright 2024 IQuestLoopCoder Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""IQuestLoopCoder model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class IQuestLoopCoderConfig(PretrainedConfig):
    r"""
    Configuration class for IQuestLoopCoder model.

    IQuestLoopCoder extends the standard LLaMA architecture with a loop mechanism:
    - Loop 1: Standard attention, stores K1, V1
    - Loop 2+: Mixed attention with gated combination of global (K1,V1) and local (K2,V2) KV

    The gate is computed as: gate = sigmoid(W @ Q + bias)
    Mixed output = gate * Attention(Q, K1, V1) + (1 - gate) * SlidingWindowAttention(Q, K2, V2)

    Args:
        vocab_size (`int`, *optional*, defaults to 76800):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27648):
            Dimension of the MLP representations (FFN hidden size).
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads (for GQA). If None, defaults to num_attention_heads.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head (hidden_size // num_attention_heads).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            Activation function in the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use past key/values for generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            Base value for rotary position embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.

        # Loop-specific parameters
        loop_num (`int`, *optional*, defaults to 2):
            Number of loops through the decoder.
        loop_window_size (`int`, *optional*, defaults to 64):
            Window size for sliding window attention in Loop 2+.
    """

    model_type = "iquestloopcoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=76800,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=80,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        # Loop-specific parameters
        loop_num=2,
        loop_window_size=64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        # GQA support
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # Loop-specific
        self.loop_num = loop_num
        self.loop_window_size = loop_window_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
