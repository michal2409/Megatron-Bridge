# Kimi Linear

[Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) is Moonshot AI's hybrid linear-attention model (48B total / 3B active parameters) built on Kimi Delta Attention (KDA). Megatron Bridge supports it through the `KimiLinearBridge`, which reuses the Kimi K3 KDA building blocks with Kimi Linear's variants and pairs them with a stock DeepSeek-V3-style MoE.

## Supported Variants

Megatron Bridge supports Kimi Linear checkpoints with the `KimiLinearForCausalLM` architecture and `kimi_linear` model type:

- Kimi-Linear-48B-A3B-Instruct: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- Kimi-Linear-48B-A3B-Base: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base

## Architecture Notes

- Hybrid attention: 3 KDA (linear-attention) layers per NoPE MLA full-attention layer, selected by the 1-based `linear_attn_config["kda_layers"]` list.
- KDA differences vs. Kimi K3: the output gate is factorized (`g_a_proj` -> `g_b_proj`) and the forget gate is computed outside the kernel via `fla.ops.kda.gate.fused_kda_gate` with no lower bound, matching the reference `modeling_kimi.py`.
- MLA differences vs. Kimi K3: direct query projection (`q_lora_rank` is null) and no output gate.
- MoE: 256 routed experts (top-8, sigmoid scoring with `e_score_correction_bias`, `routed_scaling_factor` 2.446), 1 shared expert, first layer dense. Plain SiLU-GLU experts — no K3 latent projections, SiTU, or AttnRes.
- The tokenizer is tiktoken-based (`tiktoken` and `blobfile` packages required) and declares two trailing `<|reserved_token_*|>` ids beyond the model's 163840-row embedding; training-side vocab validation tolerates that overhang and keeps the checkpoint's vocab size.

## Known Limitations

- Context parallelism is not supported for KDA layers (inherited from Kimi K3).
- Virtual pipeline parallelism is not supported.

## Related Implementation

- Bridge implementation: `src/megatron/bridge/models/kimi/kimi_linear_bridge.py`
- Attention modules: `src/megatron/bridge/models/kimi/kimi_linear_layers.py`
- Layer spec: `src/megatron/bridge/models/kimi/kimi_linear_spec.py`
