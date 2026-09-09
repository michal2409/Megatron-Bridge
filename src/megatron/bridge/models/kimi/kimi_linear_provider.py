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

"""Model provider for Kimi Linear (moonshotai/Kimi-Linear-48B-A3B)."""

from dataclasses import dataclass

from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider


@dataclass
class KimiLinearModelProvider(KimiK3ModelProvider):
    """Megatron configuration and provider for Kimi Linear.

    Kimi Linear shares the KDA/MLA hybrid layout with Kimi K3 but computes the
    KDA forget gate outside the kernel via ``fla.ops.kda.gate.fused_kda_gate``
    (no lower bound), factorizes the KDA output gate, uses a direct query
    projection in its NoPE MLA layers, and pairs it all with a stock
    DeepSeek-V3-style MoE (no latent projections, plain SiLU-GLU experts).
    """

    # unused by Kimi Linear; kept None so the KDA op takes fla's default path
    kimi_kda_gate_lower_bound: float | None = None
