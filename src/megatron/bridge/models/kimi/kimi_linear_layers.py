# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""KDA and NoPE-MLA attention for Kimi Linear.

Reuses the Kimi K3 building blocks with Kimi Linear's deltas:
  - the KDA output gate is factorized (``g_a_proj`` -> ``g_b_proj``) instead
    of K3's single ``g_proj``
  - the KDA forget gate is computed outside the kernel with
    ``fused_kda_gate`` and no lower bound, matching the reference
    ``modeling_kimi.py``
  - MLA layers use a direct query projection (``q_lora_rank`` is null) and
    have no output gate
"""

import torch
from einops import rearrange
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, KimiRMSNorm, _linear


class KimiLinearAttention(KimiK3Attention):
    """Select KDA or Kimi Linear's NoPE MLA according to the layer number."""

    def _init_kda(self, config) -> None:
        super()._init_kda(config)
        # Kimi Linear factorizes the output gate like the forget gate
        del self.g_proj
        self.g_a_proj = self._duplicated_linear(config.hidden_size, self.head_dim)
        self.g_b_proj = self._column_linear(self.head_dim, self.projection_size)

    def _init_mla(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.tp_size:
            raise ValueError(f"MLA heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim

        # q_lora_rank is null in Kimi Linear: the query is projected directly
        self.q_proj = self._column_linear(hidden_size, self.num_heads * self.q_head_dim)
        self.kv_a_proj_with_mqa = self._duplicated_linear(hidden_size, self.kv_lora_rank + self.qk_extra_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.o_proj = self._row_linear(self.num_heads * self.v_head_dim, hidden_size)

        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        from fla.ops.kda import chunk_kda
        from fla.ops.kda.gate import fused_kda_gate

        x = hidden_states.transpose(0, 1)
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if packed_seq_params is not None and cu_seqlens is None:
            raise ValueError("Packed KDA input requires cu_seqlens_q")

        conv_kwargs = {"output_final_state": False, "cu_seqlens": cu_seqlens}
        q, _ = self.q_conv1d(x=_linear(self.q_proj, x), **conv_kwargs)
        k, _ = self.k_conv1d(x=_linear(self.k_proj, x), **conv_kwargs)
        v, _ = self.v_conv1d(x=_linear(self.v_proj, x), **conv_kwargs)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        # gate computed outside the kernel, matching the reference model
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        forget_gate = fused_kda_gate(forget_gate, self.A_log, self.dt_bias)
        beta = _linear(self.b_proj, x).float().sigmoid()
        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=forget_gate,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        gate = rearrange(
            _linear(self.g_b_proj, _linear(self.g_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states
        query = _linear(self.q_proj, x)
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)
        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        key_value = _linear(self.kv_b_proj, self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        return _linear(self.o_proj, output)
