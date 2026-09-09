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

"""Hugging Face <-> Megatron conversion for Kimi Linear (48B-A3B).

Kimi Linear (https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
is the hybrid KDA / NoPE-MLA architecture that Kimi K3 builds on, paired with
a stock DeepSeek-V3-style MoE. The custom attention modules reuse the Kimi K3
implementation; the MoE path is standard MCore.

The tokenizer is tiktoken-based (requires the ``tiktoken`` and ``blobfile``
packages) and declares two trailing ``<|reserved_token_*|>`` ids beyond the
model's 163840-row embedding; the training-side vocab validation tolerates
that overhang.
"""

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_linear_provider import KimiLinearModelProvider
from megatron.bridge.models.kimi.kimi_linear_spec import build_kimi_linear_spec


@MegatronModelBridge.register_bridge(
    source="KimiLinearForCausalLM",
    target=GPTModel,
    provider=KimiLinearModelProvider,
    model_type="kimi_linear",
)
class KimiLinearBridge(MegatronModelBridge):
    """Megatron Bridge for Kimi Linear."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> KimiLinearModelProvider:
        hf_config = hf_pretrained.config
        provider = super().provider_bridge(hf_pretrained)

        provider.transformer_layer_spec = build_kimi_linear_spec
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = False
        provider.multi_latent_attention = True
        provider.position_embedding_type = "none"  # mla_use_nope
        provider.attention_backend = AttnBackend.auto

        # MLA geometry (q_lora_rank is null -> direct q projection)
        provider.q_lora_rank = None
        provider.kv_lora_rank = hf_config.kv_lora_rank
        provider.qk_head_dim = hf_config.qk_nope_head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.v_head_dim = hf_config.v_head_dim

        # MoE: DeepSeek-V3 style, no latent projections
        provider.moe_latent_size = None
        provider.num_moe_experts = hf_config.num_experts
        provider.moe_router_topk = hf_config.num_experts_per_token
        provider.moe_router_score_function = hf_config.moe_router_activation_func
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        # num_expert_group == topk_group == 1 is a no-op: disable group routing
        provider.moe_router_num_groups = None
        provider.moe_router_group_topk = None
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.num_shared_experts
        )
        provider.moe_shared_expert_overlap = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.ffn_hidden_size = hf_config.intermediate_size

        moe_layer_freq = [0] * hf_config.num_hidden_layers
        for layer_idx in range(hf_config.first_k_dense_replace, hf_config.num_hidden_layers):
            if layer_idx % getattr(hf_config, "moe_layer_freq", 1) == 0:
                moe_layer_freq[layer_idx] = 1
        provider.moe_layer_freq = moe_layer_freq

        # KDA geometry (config lists are 1-based, matching Megatron layer numbers)
        linear_config = hf_config.linear_attn_config
        provider.kimi_kda_layers = tuple(linear_config["kda_layers"])
        provider.kimi_linear_num_heads = linear_config["num_heads"]
        provider.kimi_linear_head_dim = linear_config["head_dim"]
        provider.kimi_linear_conv_kernel_size = linear_config["short_conv_kernel_size"]
        provider.kimi_kda_gate_lower_bound = None

        provider.bias_activation_fusion = False
        provider.bias_dropout_fusion = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = True
        provider.apply_rope_fusion = False
        provider.gradient_accumulation_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.should_pad_vocab = False

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map Kimi Linear's flat causal-LM layout and custom KDA parameters."""
        megatron_layer = "decoder.layers.*"
        hf_layer = "model.layers.*"
        megatron_attention = f"{megatron_layer}.self_attention"
        hf_attention = f"{hf_layer}.self_attn"

        auto_mappings = [
            ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("output_layer.weight", "lm_head.weight"),
        ]
        replicated_mappings = [
            ("decoder.final_layernorm.weight", "model.norm.weight"),
            (f"{megatron_layer}.input_layernorm.weight", f"{hf_layer}.input_layernorm.weight"),
            (f"{megatron_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.post_attention_layernorm.weight"),
            (f"{megatron_attention}.f_a_proj.weight", f"{hf_attention}.f_a_proj.weight"),
            (f"{megatron_attention}.g_a_proj.weight", f"{hf_attention}.g_a_proj.weight"),
            (f"{megatron_attention}.o_norm.weight", f"{hf_attention}.o_norm.weight"),
            (f"{megatron_attention}.kv_a_proj_with_mqa.weight", f"{hf_attention}.kv_a_proj_with_mqa.weight"),
            (f"{megatron_attention}.kv_a_layernorm.weight", f"{hf_attention}.kv_a_layernorm.weight"),
            (f"{megatron_layer}.mlp.router.weight", f"{hf_layer}.block_sparse_moe.gate.weight"),
            (
                f"{megatron_layer}.mlp.router.expert_bias",
                f"{hf_layer}.block_sparse_moe.gate.e_score_correction_bias",
            ),
        ]
        column_parallel_mappings = [
            (f"{megatron_attention}.{name}", f"{hf_attention}.{name}")
            for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "A_log",
                "dt_bias",
                "f_b_proj.weight",
                "g_b_proj.weight",
                "b_proj.weight",
                "kv_b_proj.weight",
            )
        ]
        row_parallel_mappings = [
            (f"{megatron_attention}.o_proj.weight", f"{hf_attention}.o_proj.weight"),
            (f"{megatron_layer}.mlp.linear_fc2.weight", f"{hf_layer}.mlp.down_proj.weight"),
            (
                f"{megatron_layer}.mlp.shared_experts.linear_fc2.weight",
                f"{hf_layer}.block_sparse_moe.shared_experts.down_proj.weight",
            ),
            (
                f"{megatron_layer}.mlp.experts.linear_fc2.weight*",
                f"{hf_layer}.block_sparse_moe.experts.*.w2.weight",
            ),
            (
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
                f"{hf_layer}.block_sparse_moe.experts.*.w2.weight",
            ),
        ]
        mappings = [
            *(AutoMapping(*mapping) for mapping in auto_mappings),
            *(ReplicatedMapping(*mapping) for mapping in replicated_mappings),
            *(ColumnParallelMapping(*mapping) for mapping in column_parallel_mappings),
            *(RowParallelMapping(*mapping) for mapping in row_parallel_mappings),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.gate_proj.weight",
                up=f"{hf_layer}.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{hf_layer}.block_sparse_moe.shared_experts.gate_proj.weight",
                up=f"{hf_layer}.block_sparse_moe.shared_experts.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.linear_fc1.weight*",
                gate=f"{hf_layer}.block_sparse_moe.experts.*.w1.weight",
                up=f"{hf_layer}.block_sparse_moe.experts.*.w3.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                gate=f"{hf_layer}.block_sparse_moe.experts.*.w1.weight",
                up=f"{hf_layer}.block_sparse_moe.experts.*.w3.weight",
            ),
        ]
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Flatten A_log from HF's ``(1, 1, num_heads, 1)`` to Megatron's ``(num_heads,)``."""
        if isinstance(hf_param, dict):
            return {key: hf_state_dict[name] for key, name in hf_param.items()}
        weight = hf_state_dict[hf_param]
        if hf_param.endswith(".self_attn.A_log"):
            weight = weight.reshape(-1)
        return weight

    def maybe_modify_converted_hf_weight(self, task, converted_weights_dict, hf_state_dict):
        """Restore A_log to HF's ``(1, 1, num_heads, 1)`` layout on export."""
        result = {}
        for name, weight in converted_weights_dict.items():
            if name.endswith(".self_attn.A_log") and weight.ndim == 1:
                weight = weight.view(1, 1, -1, 1)
            result[name] = weight
        return result
