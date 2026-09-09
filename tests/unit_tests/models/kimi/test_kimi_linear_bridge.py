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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_linear_bridge import KimiLinearBridge
from megatron.bridge.models.kimi.kimi_linear_provider import KimiLinearModelProvider


@pytest.fixture
def kimi_linear_config() -> SimpleNamespace:
    """The official Kimi-Linear-48B-A3B architecture truncated to four layers."""
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        first_k_dense_replace=1,
        head_dim=72,
        hidden_act="silu",
        hidden_size=2304,
        initializer_range=0.02,
        intermediate_size=9216,
        kv_lora_rank=512,
        linear_attn_config={
            "full_attn_layers": [4],
            "head_dim": 128,
            "kda_layers": [1, 2, 3],
            "num_heads": 32,
            "short_conv_kernel_size": 4,
        },
        mla_use_nope=True,
        model_max_length=1048576,
        moe_intermediate_size=1024,
        moe_layer_freq=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_attention_heads=32,
        num_expert_group=1,
        num_experts=256,
        num_experts_per_token=8,
        num_hidden_layers=4,
        num_key_value_heads=32,
        num_nextn_predict_layers=0,
        num_shared_experts=1,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-5,
        rope_scaling=None,
        rope_theta=10000.0,
        routed_scaling_factor=2.446,
        tie_word_embeddings=False,
        topk_group=1,
        torch_dtype="bfloat16",
        use_grouped_topk=True,
        v_head_dim=128,
        vocab_size=163840,
    )


@pytest.fixture
def kimi_linear_pretrained(kimi_linear_config: SimpleNamespace) -> Mock:
    """Return a config-only Kimi Linear wrapper."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = kimi_linear_config
    return pretrained


def test_provider_bridge_configures_four_layer_proxy(kimi_linear_pretrained: Mock) -> None:
    """The provider preserves the KDA/MLA layout with a stock DSv3-style MoE."""
    provider = KimiLinearBridge().provider_bridge(kimi_linear_pretrained)

    assert isinstance(provider, KimiLinearModelProvider)
    assert provider.num_layers == 4
    assert provider.position_embedding_type == "none"
    assert provider.kimi_kda_layers == (1, 2, 3)
    assert provider.kimi_linear_num_heads == 32
    assert provider.kimi_linear_head_dim == 128
    assert provider.kimi_kda_gate_lower_bound is None
    assert provider.moe_layer_freq == [0, 1, 1, 1]
    # direct query projection and no latent MoE
    assert provider.q_lora_rank is None
    assert provider.moe_latent_size is None
    assert provider.kv_lora_rank == 512
    assert provider.qk_head_dim == 128
    assert provider.qk_pos_emb_head_dim == 64
    assert provider.num_moe_experts == 256
    assert provider.moe_ffn_hidden_size == 1024
    assert provider.moe_shared_expert_intermediate_size == 1024
    assert provider.moe_router_topk == 8
    assert provider.moe_router_score_function == "sigmoid"
    assert provider.moe_router_topk_scaling_factor == pytest.approx(2.446)
    # single-group top-k is a no-op and stays disabled
    assert provider.moe_router_num_groups is None
    assert provider.moe_router_group_topk is None
    assert provider.hidden_dropout == 0.0
    assert provider.attention_dropout == 0.0
    assert provider.bf16 is True
    assert provider.params_dtype == torch.bfloat16


def test_mapping_registry_covers_kda_and_stock_moe(kimi_linear_pretrained: Mock) -> None:
    """Custom KDA weights and the flat MoE layout resolve in both directions."""
    bridge = KimiLinearBridge()
    bridge.provider_bridge(kimi_linear_pretrained)
    registry = bridge.mapping_registry()

    cases = {
        "decoder.layers.0.self_attention.g_a_proj.weight": (
            "model.layers.0.self_attn.g_a_proj.weight",
            ReplicatedMapping,
        ),
        "decoder.layers.0.self_attention.g_b_proj.weight": (
            "model.layers.0.self_attn.g_b_proj.weight",
            ColumnParallelMapping,
        ),
        "decoder.layers.1.self_attention.q_conv1d.weight": (
            "model.layers.1.self_attn.q_conv1d.weight",
            ColumnParallelMapping,
        ),
        "decoder.layers.3.self_attention.o_proj.weight": (
            "model.layers.3.self_attn.o_proj.weight",
            RowParallelMapping,
        ),
        "decoder.layers.2.mlp.router.expert_bias": (
            "model.layers.2.block_sparse_moe.gate.e_score_correction_bias",
            ReplicatedMapping,
        ),
    }
    for megatron_name, (hf_name, mapping_type) in cases.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert isinstance(mapping, mapping_type)
        assert mapping.hf_param == hf_name
        reverse = registry.hf_to_megatron_lookup(hf_name)
        assert reverse is not None
        assert reverse.megatron_param == megatron_name

    fc1 = registry.megatron_to_hf_lookup("decoder.layers.2.mlp.experts.linear_fc1.weight5")
    assert isinstance(fc1, GatedMLPMapping)

    # Kimi Linear has no K3-only modules
    assert registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.g_proj.weight") is None
    assert registry.megatron_to_hf_lookup("decoder.layers.2.mlp.fc1_latent_proj.weight") is None


def test_a_log_import_flattens_hf_layout(kimi_linear_pretrained: Mock) -> None:
    """A_log imports from HF's (1, 1, H, 1) tensor to Megatron's (H,)."""
    bridge = KimiLinearBridge()
    bridge.hf_config = kimi_linear_pretrained.config
    name = "model.layers.0.self_attn.A_log"
    source = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1)

    result = bridge.maybe_modify_loaded_hf_weight(name, {name: source})

    assert result.shape == (32,)
    torch.testing.assert_close(result, torch.arange(32, dtype=torch.float32), rtol=0, atol=0)


def test_a_log_export_restores_hf_layout() -> None:
    """A_log exports back to HF's (1, 1, H, 1) layout."""
    bridge = KimiLinearBridge()
    name = "model.layers.0.self_attn.A_log"
    active = torch.arange(32, dtype=torch.float32)
    task = SimpleNamespace(weight_dtype=None)

    result = bridge.maybe_modify_converted_hf_weight(task, {name: active}, {name: active.view(1, 1, 32, 1)})

    assert result[name].shape == (1, 1, 32, 1)
    torch.testing.assert_close(result[name].reshape(-1), active, rtol=0, atol=0)
