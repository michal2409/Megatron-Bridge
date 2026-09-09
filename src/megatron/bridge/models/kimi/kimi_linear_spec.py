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

"""MCore layer specification for Kimi Linear."""

import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.kimi.kimi_linear_layers import KimiLinearAttention


def build_kimi_linear_spec(config, vp_stage=None):
    """Build the heterogeneous KDA/MLA block with stock dense/MoE layers."""
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Kimi Linear does not support virtual pipeline parallelism yet")

    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.submodules.self_attention = ModuleSpec(module=KimiLinearAttention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm
        if not config.moe_layer_freq[layer_offset + len(layer_specs)]:
            layer_spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        layer_specs.append(layer_spec)
    block_spec.layer_specs = layer_specs
    return block_spec
