"""
Modified MIT License

Software Copyright© 2025 IQuest Research

Our only modification is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services, you shall
prominently display "IQuest Coder" on the user interface of such product or
service.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_iquestloopcoder import IQuestLoopCoderConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "IQuestLoopCoderConfig"


class IQuestLoopCoderCache(Cache):
    """Cache implementation for IQuestLoopCoder that manages shared and local KV caches.

    - shared_key_cache/shared_value_cache: Stores KV from Loop 1 (global context)
    - local_key_cache/local_value_cache: Stores KV from Loop 2+ (local window, only window_size tokens)

    Multi-GPU Support:
    - Each layer's cache is stored on the same device as that layer's output
    - Device is tracked per-layer to handle cross-device scenarios
    """

    def __init__(self, window_size: int, num_layers: int):
        # We intentionally don't call super().__init__ because the parent assumes static cache sizes.
        self.window_size = window_size
        self.num_layers = num_layers

        # Shared cache: stores Loop 1 KV (global context)
        self.shared_key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.shared_value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        # Local cache: stores Loop 2+ KV (sliding window, only window_size tokens)
        self.local_key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.local_value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        self.layers: List[Any] = []  # attribute expected by HF Cache utilities
        self._seen_tokens = 0

    def _ensure_same_device(self, tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Ensure tensor is on the target device (for multi-GPU support)."""
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor

    def update_shared(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update shared cache (Loop 1 KV)."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")

        cached_key = self.shared_key_cache[layer_idx]
        cached_value = self.shared_value_cache[layer_idx]

        if cached_key is None:
            self.shared_key_cache[layer_idx] = key_states
            self.shared_value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            # Multi-GPU: ensure cached tensors are on the same device as incoming tensors
            cached_key = self._ensure_same_device(cached_key, key_states.device)
            cached_value = self._ensure_same_device(cached_value, value_states.device)
            self.shared_key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.shared_value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.shared_key_cache[layer_idx]
        result_value = self.shared_value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        # Track sequence length
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def update_local(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update local cache (Loop 2+ KV) with sliding window management.

        If the cache is full (window_size tokens), remove the oldest token and add the new one.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")

        cached_key = self.local_key_cache[layer_idx]
        cached_value = self.local_value_cache[layer_idx]

        if cached_key is None:
            # First token in local cache
            self.local_key_cache[layer_idx] = key_states
            self.local_value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None

            # Multi-GPU: ensure cached tensors are on the same device as incoming tensors
            cached_key = self._ensure_same_device(cached_key, key_states.device)
            cached_value = self._ensure_same_device(cached_value, value_states.device)

            # Check if we need to remove the oldest token
            current_len = cached_key.shape[2]
            if current_len >= self.window_size:
                # Remove the first token (oldest) and add the new one
                self.local_key_cache[layer_idx] = torch.cat([cached_key[:, :, 1:, :], key_states], dim=2)
                self.local_value_cache[layer_idx] = torch.cat([cached_value[:, :, 1:, :], value_states], dim=2)
            else:
                # Just append
                self.local_key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
                self.local_value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.local_key_cache[layer_idx]
        result_value = self.local_value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        return result_key, result_value

    def get_shared(
        self, layer_idx: int, target_device: Optional[torch.device] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get shared cache for a layer.

        Args:
            layer_idx: The layer index
            target_device: If provided, move cache to this device (for multi-GPU)
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return None, None
        key = self.shared_key_cache[layer_idx]
        value = self.shared_value_cache[layer_idx]
        if target_device is not None and key is not None and value is not None:
            key = self._ensure_same_device(key, target_device)
            value = self._ensure_same_device(value, target_device)
        return key, value

    def get_local(
        self, layer_idx: int, target_device: Optional[torch.device] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get local cache for a layer.

        Args:
            layer_idx: The layer index
            target_device: If provided, move cache to this device (for multi-GPU)
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return None, None
        key = self.local_key_cache[layer_idx]
        value = self.local_value_cache[layer_idx]
        if target_device is not None and key is not None and value is not None:
            key = self._ensure_same_device(key, target_device)
            value = self._ensure_same_device(value, target_device)
        return key, value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default update method (for compatibility, updates shared cache)."""
        return self.update_shared(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length from shared cache."""
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or layer_idx >= len(self.shared_key_cache):
            return 0
        cached = self.shared_key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search."""
        for layer_idx in range(self.num_layers):
            if self.shared_key_cache[layer_idx] is not None:
                device = self.shared_key_cache[layer_idx].device
                self.shared_key_cache[layer_idx] = self.shared_key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.shared_value_cache[layer_idx] = self.shared_value_cache[layer_idx].index_select(0, beam_idx.to(device))

            if self.local_key_cache[layer_idx] is not None:
                device = self.local_key_cache[layer_idx].device
                self.local_key_cache[layer_idx] = self.local_key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.local_value_cache[layer_idx] = self.local_value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        """Clear all caches."""
        logger.debug("Clearing IQuestLoopCoderCache")
        self.shared_key_cache = [None] * self.num_layers
        self.shared_value_cache = [None] * self.num_layers
        self.local_key_cache = [None] * self.num_layers
        self.local_value_cache = [None] * self.num_layers
        self._seen_tokens = 0


class IQuestLoopCoderRMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class IQuestLoopCoderRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim, max_position_embeddings=8192, base=500000.0, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        # Multi-GPU: ensure all tensors are on the same device as input x
        target_device = x.device

        inv_freq = self.inv_freq
        if inv_freq.device != target_device:
            inv_freq = inv_freq.to(target_device)

        # Ensure position_ids is on the same device
        if position_ids.device != target_device:
            position_ids = position_ids.to(target_device)

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = target_device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class IQuestLoopCoderMLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LoopGateProjection(nn.Module):
    """Gate projection for mixed attention in Loop 2+.

    Computes: g = sigmoid(linear(Q)) for each head independently.
    This gate determines how much to use Loop1's KV (global) vs current loop's KV (local).
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Each head has its own gate: Linear(head_dim -> 1) per head
        # Implemented as [num_heads, head_dim] weight + [num_heads] bias
        self.weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Compute gate values from query tensor.

        Args:
            query: [batch, num_heads, seq_len, head_dim]

        Returns:
            gate: [batch, num_heads, seq_len, 1]
        """
        # query: [batch, num_heads, seq_len, head_dim]
        # weight: [num_heads, head_dim]
        # Multi-GPU: ensure weight/bias are on the same device as query
        weight = self.weight
        bias = self.bias
        if weight.device != query.device:
            weight = weight.to(query.device)
            bias = bias.to(query.device)

        # For each head h: gate_h = query[:, h, :, :] @ weight[h, :].T + bias[h]
        # Using einsum: gate = einsum('bhsd,hd->bhs', query, weight) + bias
        gate_logits = torch.einsum('bhsd,hd->bhs', query, weight)  # [batch, num_heads, seq_len]
        gate_logits = gate_logits + bias[None, :, None]  # broadcast bias
        gate = torch.sigmoid(gate_logits)
        return gate.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]


class IQuestLoopCoderAttention(nn.Module):
    """Multi-head attention with GQA support."""

    def __init__(self, config: IQuestLoopCoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = IQuestLoopCoderRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

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
        device = hidden_states.device

        # Multi-GPU: ensure attention_mask is on the same device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def forward_with_external_kv(
        self,
        hidden_states: torch.Tensor,
        external_key: torch.Tensor,
        external_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass using external K, V (for Loop 2+ mixed attention).

        Args:
            hidden_states: Input for computing Q
            external_key: Pre-computed K (already with RoPE applied)
            external_value: Pre-computed V
            attention_mask: Causal attention mask
            position_ids: Position IDs
            sliding_window: If set, apply sliding window attention

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # Multi-GPU: ensure attention_mask is on the same device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        # Compute Q from current hidden states
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Use external K, V (already have RoPE for K)
        # Multi-GPU: ensure external KV are on the same device as query
        key_states = external_key
        value_states = external_value
        if key_states.device != query_states.device:
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask (causal)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sliding window mask if needed
        if sliding_window is not None and q_len > sliding_window:
            # Create sliding window mask
            # For each position i, can only attend to [i-window+1, i]
            seq_len = key_states.shape[2]
            row_idx = torch.arange(q_len, device=query_states.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=query_states.device).unsqueeze(0)
            window_mask = (col_idx > row_idx) | (col_idx < row_idx - sliding_window + 1)
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, seq_len]
            attn_weights = attn_weights.masked_fill(window_mask, float('-inf'))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Don't apply o_proj here - return raw attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output  # [batch, seq_len, num_heads, head_dim]

    def get_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get Q, K, V tensors with RoPE applied.

        Returns:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_kv_heads, seq_len, head_dim]
            value: [batch, num_kv_heads, seq_len, head_dim]
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def forward_decode_loop1(
        self,
        hidden_states: torch.Tensor,
        past_shared_key: Optional[torch.Tensor],
        past_shared_value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for Loop 1 in decode stage.

        Args:
            hidden_states: Current hidden states [batch, 1, hidden_size]
            past_shared_key: Past shared keys from cache [batch, num_kv_heads, past_len, head_dim]
            past_shared_value: Past shared values from cache [batch, num_kv_heads, past_len, head_dim]
            attention_mask: Causal attention mask
            position_ids: Position IDs
            cache_position: Cache position

        Returns:
            output: Attention output [batch, 1, hidden_size]
            k1: Current key [batch, num_kv_heads, 1, head_dim] (only current token)
            v1: Current value [batch, num_kv_heads, 1, head_dim] (only current token)
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # Multi-GPU: ensure attention_mask is on the same device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Store current token's k1, v1 for return (before concatenation)
        k1_current = key_states  # [batch, num_kv_heads, 1, head_dim]
        v1_current = value_states  # [batch, num_kv_heads, 1, head_dim]

        # Concatenate with past shared KV cache for attention computation
        if past_shared_key is not None and past_shared_value is not None:
            # Multi-GPU: ensure past KV are on the same device
            if past_shared_key.device != key_states.device:
                past_shared_key = past_shared_key.to(key_states.device)
                past_shared_value = past_shared_value.to(value_states.device)
            key_states = torch.cat([past_shared_key, key_states], dim=2)
            value_states = torch.cat([past_shared_value, value_states], dim=2)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, k1_current, v1_current

    def forward_decode_loop2(
        self,
        hidden_states: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        past_shared_key: Optional[torch.Tensor],
        past_shared_value: Optional[torch.Tensor],
        past_local_key: Optional[torch.Tensor],
        past_local_value: Optional[torch.Tensor],
        gate_proj: LoopGateProjection,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loop_window_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for Loop 2 in decode stage with mixed attention.

        Args:
            hidden_states: Current hidden states [batch, 1, hidden_size]
            k1: Key from Loop 1 (current token) [batch, num_kv_heads, 1, head_dim]
            v1: Value from Loop 1 (current token) [batch, num_kv_heads, 1, head_dim]
            past_shared_key: Past shared keys from cache [batch, num_kv_heads, past_len, head_dim]
            past_shared_value: Past shared values from cache [batch, num_kv_heads, past_len, head_dim]
            past_local_key: Past local keys from cache [batch, num_kv_heads, window_len, head_dim]
            past_local_value: Past local values from cache [batch, num_kv_heads, window_len, head_dim]
            gate_proj: Gate projection module
            attention_mask: Causal attention mask
            position_ids: Position IDs
            loop_window_size: Window size for sliding window attention

        Returns:
            output: Attention output [batch, 1, hidden_size]
            k2: Current key [batch, num_kv_heads, 1, head_dim]
            v2: Current value [batch, num_kv_heads, 1, head_dim]
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # Multi-GPU: ensure attention_mask is on the same device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        # Get Q2, K2, V2 for current loop
        q2, k2, v2 = self.get_qkv(hidden_states, position_ids)

        # Compute gate: g = sigmoid(linear(Q2))
        gate = gate_proj(q2)  # [batch, num_heads, 1, 1]

        # Multi-GPU: ensure k1, v1 are on the same device
        if k1.device != device:
            k1 = k1.to(device)
            v1 = v1.to(device)

        # For attention A: concatenate past shared KV with current k1, v1 (full global context)
        if past_shared_key is not None and past_shared_value is not None:
            if past_shared_key.device != device:
                past_shared_key = past_shared_key.to(device)
                past_shared_value = past_shared_value.to(device)
            k1_full = torch.cat([past_shared_key, k1], dim=2)
            v1_full = torch.cat([past_shared_value, v1], dim=2)
        else:
            k1_full = k1
            v1_full = v1

        # For attention B: concatenate past local KV with current k2, v2 (sliding window)
        if past_local_key is not None and past_local_value is not None:
            if past_local_key.device != device:
                past_local_key = past_local_key.to(device)
                past_local_value = past_local_value.to(device)
            k2_full = torch.cat([past_local_key, k2], dim=2)
            v2_full = torch.cat([past_local_value, v2], dim=2)
        else:
            k2_full = k2
            v2_full = v2

        # Repeat KV for GQA
        k1_expanded = repeat_kv(k1_full, self.num_key_value_groups)
        v1_expanded = repeat_kv(v1_full, self.num_key_value_groups)
        k2_expanded = repeat_kv(k2_full, self.num_key_value_groups)
        v2_expanded = repeat_kv(v2_full, self.num_key_value_groups)

        # Attention A: Q2 @ K1_full, V1_full (global, full sequence)
        head_dim = q2.shape[-1]
        attn_weights_A = torch.matmul(q2, k1_expanded.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k1_expanded.shape[-2]]
            attn_weights_A = attn_weights_A + causal_mask
        attn_weights_A = nn.functional.softmax(attn_weights_A, dim=-1, dtype=torch.float32).to(q2.dtype)
        attn_A = torch.matmul(attn_weights_A, v1_expanded)

        # Attention B: Q2 @ K2_full, V2_full (local sliding window)
        attn_weights_B = torch.matmul(q2, k2_expanded.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k2_expanded.shape[-2]]
            attn_weights_B = attn_weights_B + causal_mask

        # Apply sliding window mask
        q_len_attn = q2.shape[2]
        k_len_attn = k2_expanded.shape[2]
        if q_len_attn <= loop_window_size:
            # If sequence fits in window, use standard attention
            attn_weights_B = nn.functional.softmax(attn_weights_B, dim=-1, dtype=torch.float32).to(q2.dtype)
        else:
            # Apply sliding window mask
            row_idx = torch.arange(q_len_attn, device=device).unsqueeze(1)
            col_idx = torch.arange(k_len_attn, device=device).unsqueeze(0)
            window_mask = (col_idx > row_idx) | (col_idx < row_idx - loop_window_size + 1)
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            attn_weights_B = attn_weights_B.masked_fill(window_mask, float('-inf'))
            attn_weights_B = nn.functional.softmax(attn_weights_B, dim=-1, dtype=torch.float32).to(q2.dtype)
        attn_B = torch.matmul(attn_weights_B, v2_expanded)

        # Mixed attention: gate * A + (1 - gate) * B
        mixed_attn = gate * attn_A + (1 - gate) * attn_B

        # Reshape and apply output projection
        bsz, num_heads, seq_len, head_dim = mixed_attn.shape
        mixed_attn = mixed_attn.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(mixed_attn)

        return attn_output, k2, v2


class IQuestLoopCoderDecoderLayer(nn.Module):
    """Transformer decoder layer with integrated gate projection for multi-GPU support."""

    def __init__(self, config: IQuestLoopCoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = IQuestLoopCoderAttention(config=config, layer_idx=layer_idx)
        self.mlp = IQuestLoopCoderMLP(config)
        self.input_layernorm = IQuestLoopCoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = IQuestLoopCoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gate projection integrated into decoder layer for multi-GPU support
        # This ensures the gate is on the same device as the layer
        self.gate_projection = LoopGateProjection(config.num_attention_heads, config.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def forward_loop2_mixed(
        self,
        hidden_states: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loop_window_size: int = 64,
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass for Loop 2+ with mixed attention.

        Uses the integrated gate_projection for multi-GPU compatibility.

        Args:
            hidden_states: Current hidden states
            k1: Key from Loop 1 [batch, num_kv_heads, seq_len, head_dim]
            v1: Value from Loop 1 [batch, num_kv_heads, seq_len, head_dim]
            attention_mask: Causal attention mask
            position_ids: Position IDs
            loop_window_size: Window size for sliding window attention

        Returns:
            output hidden states, gate mean value
        """
        device = hidden_states.device

        # Multi-GPU: ensure attention_mask is on the same device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        residual = hidden_states
        hidden_states_normed = self.input_layernorm(hidden_states)

        # Get Q2, K2, V2 for current loop
        q2, k2, v2 = self.self_attn.get_qkv(hidden_states_normed, position_ids)

        # Multi-GPU: ensure k1, v1 are on the same device as current layer
        if k1.device != device:
            k1 = k1.to(device)
            v1 = v1.to(device)

        # Compute gate using integrated gate_projection (same device as layer)
        gate = self.gate_projection(q2)  # [batch, num_heads, seq_len, 1]
        gate_mean = gate.detach().mean().item()

        # Repeat K1, V1 for GQA
        k1_expanded = repeat_kv(k1, self.self_attn.num_key_value_groups)
        v1_expanded = repeat_kv(v1, self.self_attn.num_key_value_groups)
        k2_expanded = repeat_kv(k2, self.self_attn.num_key_value_groups)
        v2_expanded = repeat_kv(v2, self.self_attn.num_key_value_groups)

        # Attention A: Q2 @ K1, V1 (global, full sequence)
        attn_A = self._compute_attention(q2, k1_expanded, v1_expanded, attention_mask)

        # Attention B: Q2 @ K2, V2 (local sliding window)
        attn_B = self._compute_attention_with_window(q2, k2_expanded, v2_expanded, attention_mask, loop_window_size)

        # Mixed attention: gate * A + (1 - gate) * B
        # attn_A, attn_B: [batch, num_heads, seq_len, head_dim]
        mixed_attn = gate * attn_A + (1 - gate) * attn_B

        # Reshape and apply output projection
        bsz, num_heads, seq_len, head_dim = mixed_attn.shape
        mixed_attn = mixed_attn.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        hidden_states = self.self_attn.o_proj(mixed_attn)

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, gate_mean

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard attention computation."""
        head_dim = query.shape[-1]
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _compute_attention_with_window(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        window_size: int,
    ) -> torch.Tensor:
        """Attention with sliding window."""
        q_len = query.shape[2]
        k_len = key.shape[2]
        head_dim = query.shape[-1]

        # If sequence fits in window, use standard attention
        if q_len <= window_size:
            return self._compute_attention(query, key, value, attention_mask)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

        # Apply causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sliding window mask
        row_idx = torch.arange(q_len, device=query.device).unsqueeze(1)
        col_idx = torch.arange(k_len, device=query.device).unsqueeze(0)
        # Can only attend to positions in [i - window_size + 1, i]
        window_mask = (col_idx > row_idx) | (col_idx < row_idx - window_size + 1)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(window_mask, float('-inf'))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output


class IQuestLoopCoderPreTrainedModel(PreTrainedModel):
    """Base class for IQuestLoopCoder models."""
    config_class = IQuestLoopCoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IQuestLoopCoderDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class IQuestLoopCoderModel(IQuestLoopCoderPreTrainedModel):
    """IQuestLoopCoder Transformer decoder model."""

    def __init__(self, config: IQuestLoopCoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            IQuestLoopCoderDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = IQuestLoopCoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Loop configuration
        self.loop_num = config.loop_num
        self.loop_window_size = config.loop_window_size

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_length = inputs_embeds.shape[1]

        # Determine which forward path to use:
        # 1. If past_key_values exists and seq_length == 1: autoregressive generation step
        #    -> Use standard attention with KV cache (no loop needed for single token)
        # 2. Otherwise (prefill or training): use loop mechanism

        is_generation_step = past_key_values is not None and seq_length == 1

        if is_generation_step:
            # Autoregressive generation: single token, use KV cache
            return self._forward_with_cache(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        # Prefill or training: use loop mechanism
        return self._forward_loop(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    def _forward_loop(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward with loop mechanism (for training and prefill).

        This implements the Loop mechanism:
        - Loop 1: Standard attention, stores K1, V1 for each layer
        - Loop 2+: Mixed attention with gated combination of global (K1,V1) and local (K2,V2)
        """
        batch_size, seq_length, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)

        if cache_position is None:
            cache_position = torch.arange(seq_length, device=device)

        # Create causal mask
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, output_attentions)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # For KV cache during prefill - use IQuestLoopCoderCache
        if use_cache:
            next_decoder_cache = IQuestLoopCoderCache(self.loop_window_size, len(self.layers))
        else:
            next_decoder_cache = None

        # Store K1, V1 for each layer (for Loop 2+ access)
        # Note: In multi-GPU, each layer's K1/V1 will be on that layer's device
        layer_k1v1: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(len(self.layers))
        ]

        # ============ Loop 1: Standard forward, store K1, V1 ============
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get K1, V1 before standard forward (from original hidden_states, after layernorm)
            hidden_states_normed = decoder_layer.input_layernorm(hidden_states)
            q1, k1, v1 = decoder_layer.self_attn.get_qkv(hidden_states_normed, position_ids)

            # Store K1, V1 for Loop 2+ (on the same device as this layer)
            layer_k1v1[layer_idx] = (k1, v1)

            # Store K1, V1 in shared cache
            if use_cache:
                next_decoder_cache.update_shared(k1, v1, layer_idx)

            # Standard forward
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # ============ Loop 2 to loop_num: Mixed attention ============
        for loop_idx in range(2, self.loop_num + 1):
            for layer_idx, decoder_layer in enumerate(self.layers):
                # Get K1, V1 stored from Loop 1
                k1, v1 = layer_k1v1[layer_idx]
                if k1 is None or v1 is None:
                    # Fallback: compute K1, V1 if not stored
                    hidden_states_normed = decoder_layer.input_layernorm(hidden_states)
                    _, k1, v1 = decoder_layer.self_attn.get_qkv(hidden_states_normed, position_ids)

                # Use integrated gate_projection in decoder layer
                hidden_states, gate_mean = decoder_layer.forward_loop2_mixed(
                    hidden_states,
                    k1=k1,
                    v1=v1,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    loop_window_size=self.loop_window_size,
                )

                # Store Loop 2+ KV in local cache (only for loop_idx == 2)
                if use_cache and loop_idx == 2:
                    hidden_states_normed = decoder_layer.input_layernorm(hidden_states)
                    _, k2, v2 = decoder_layer.self_attn.get_qkv(hidden_states_normed, position_ids)
                    next_decoder_cache.update_local(k2, v2, layer_idx)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _forward_with_cache(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        cache_position: Optional[torch.LongTensor],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward with KV cache using loop mechanism (for inference generation).

        Loop 1: Standard attention, uses shared KV cache (previous tokens + current token)
        Loop 2+: Mixed attention, uses local KV cache (sliding window)
        """
        batch_size, seq_length, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        # Ensure we're using IQuestLoopCoderCache
        if use_cache:
            if not isinstance(past_key_values, IQuestLoopCoderCache):
                # Convert to IQuestLoopCoderCache if needed
                next_decoder_cache = IQuestLoopCoderCache(self.loop_window_size, len(self.layers))
                # Copy existing cache if possible
                if past_key_values is not None:
                    for layer_idx in range(len(self.layers)):
                        try:
                            past_k = past_key_values.key_cache[layer_idx] if hasattr(past_key_values, 'key_cache') else None
                            past_v = past_key_values.value_cache[layer_idx] if hasattr(past_key_values, 'value_cache') else None
                            if past_k is not None and past_v is not None:
                                next_decoder_cache.update_shared(past_k, past_v, layer_idx)
                        except:
                            pass
            else:
                next_decoder_cache = past_key_values
        else:
            next_decoder_cache = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Store Loop 1 K1, V1 for each layer (for Loop 2+ access)
        loop1_k1v1: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(len(self.layers))
        ]

        # ============ Loop 1: Standard attention, store in shared cache ============
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get past shared KV cache (move to current layer's device)
            layer_device = next(decoder_layer.parameters()).device
            past_shared_key, past_shared_value = None, None
            if next_decoder_cache is not None:
                past_shared_key, past_shared_value = next_decoder_cache.get_shared(layer_idx, target_device=layer_device)

            # Forward Loop 1
            attn_output, k1, v1 = decoder_layer.self_attn.forward_decode_loop1(
                hidden_states=decoder_layer.input_layernorm(hidden_states),
                past_shared_key=past_shared_key,
                past_shared_value=past_shared_value,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )

            # Store current token's K1, V1 for Loop 2+
            loop1_k1v1[layer_idx] = (k1, v1)

            # Update shared cache with current token's Loop 1 KV
            if use_cache:
                next_decoder_cache.update_shared(k1, v1, layer_idx)

            hidden_states = hidden_states + attn_output

            # MLP
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if output_attentions:
                all_self_attns += (None,)  # We don't return attention weights in decode loop

        # ============ Loop 2 to loop_num: Mixed attention ============
        for loop_idx in range(2, self.loop_num + 1):
            for layer_idx, decoder_layer in enumerate(self.layers):
                layer_device = next(decoder_layer.parameters()).device

                # Get k1, v1 (current token's Loop 1 KV)
                k1_current, v1_current = loop1_k1v1[layer_idx]
                if k1_current is None or v1_current is None:
                    continue

                # Get full shared KV and past local KV from cache
                k1_full, v1_full = None, None
                past_local_key, past_local_value = None, None
                if next_decoder_cache is not None:
                    k1_full, v1_full = next_decoder_cache.get_shared(layer_idx, target_device=layer_device)
                    past_local_key, past_local_value = next_decoder_cache.get_local(layer_idx, target_device=layer_device)

                # Use integrated gate_projection in decoder layer
                attn_output, k2, v2 = decoder_layer.self_attn.forward_decode_loop2(
                    hidden_states=decoder_layer.input_layernorm(hidden_states),
                    k1=k1_current,
                    v1=v1_current,
                    past_shared_key=k1_full[:, :, :-1, :] if k1_full is not None and k1_full.shape[2] > 1 else None,
                    past_shared_value=v1_full[:, :, :-1, :] if v1_full is not None and v1_full.shape[2] > 1 else None,
                    past_local_key=past_local_key,
                    past_local_value=past_local_value,
                    gate_proj=decoder_layer.gate_projection,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    loop_window_size=self.loop_window_size,
                )

                # Update local cache with current token's Loop 2+ KV
                if use_cache and loop_idx == 2:
                    next_decoder_cache.update_local(k2, v2, layer_idx)

                hidden_states = hidden_states + attn_output

                # MLP
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                hidden_states = decoder_layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """Create causal attention mask."""
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        # Determine target length for attention
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            target_length = past_length + sequence_length
        elif attention_mask is not None:
            target_length = attention_mask.shape[-1]
        else:
            target_length = sequence_length

        # Create causal mask
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            # For prefill: standard causal mask
            causal_mask = torch.triu(causal_mask, diagonal=1)

        # Adjust for cache position (for generation steps after prefill)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            if mask_length <= target_length:
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask


class IQuestLoopCoderForCausalLM(IQuestLoopCoderPreTrainedModel, GenerationMixin):
    """IQuestLoopCoder model with a causal language modeling head."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = IQuestLoopCoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_ids.shape[1]:]

        position_ids = cache_position.unsqueeze(0)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
