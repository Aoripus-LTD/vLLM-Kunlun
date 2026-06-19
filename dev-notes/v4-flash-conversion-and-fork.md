# V4 Flash: Conversion & Fork Integration

## Status: CONVERSION ✅ | SERVE ❌ (architectural incompatibility)

## What Worked

### 1. FP8/FP4 → BF16 Conversion (commit 7ea56b0, a26e878)
- **149GB source → 543GB BF16 output** (46/46 shards, 35020 keys)
- 34167 FP8/FP4 weights dequantized to BF16
- 853 BF16 pass-through (norms, biases, embed, lm_head)
- 34167 scale entries removed from index
- Runtime: ~45 min on CPU
- **Hand-parsed safetensors** to bypass container torch's missing `float8_e8m0fnu`:
  - safetensors 0.7.0 `PySafeSlice` has only `get_dtype`/`get_shape` (no `get_data`)
  - `framework="np"` fails: `"bfloat16 not understood"`
  - `framework="pt"` triggers `torch.float8_e8m0fnu` lookup that older torch lacks
  - Solution: read `[8B LE uint64 header_len][JSON header][raw data]` directly
  - `torch.frombuffer(raw, dtype=SAFE_DTYPE[dtype_str]).reshape(shape)`

### 2. Fork Whitelist (commit f44dddb)
- Added `"deepseek_v4"` to `is_deepseek_mla` whitelist in `vllm_kunlun/config/model.py`
- `DeepseekV4ForCausalLM` was already registered in `models/__init__.py:131` → maps to `DeepseekV3ForCausalLM`

### 3. Config Cleanup
- Removed `auto_map` (would try to load missing `DeepseekV4ForCausalLM` custom code)
- Removed `quantization_config` (we are BF16, no quant)
- Changed `model_type: "deepseek_v4"` → `"deepseek_v3"` (Transformers recognizes V3)
- Set `architectures: ["DeepseekV4ForCausalLM"]` (routes to fork's V3 code)

## What's Blocked

**V4 architecture is fundamentally incompatible with fork's V3 code.**

V4 introduces (per DeepSeek V4 paper + HF transformers v5.8.0 docs):
- **Manifold-Constrained Hyper-Connections (mHC)**: `hc_mult=4` parallel residual streams
  - Weights: `hc_attn_base`, `hc_attn_fn`, `hc_attn_scale`, `hc_ffn_*`, `hc_head_*`
- **Compressed Sparse Attention (CSA)**: compress KV 4x + Lightning Indexer
- **Heavily Compressed Attention (HCA)**: compress KV 128x, dense on compressed
- **Lightning Indexer**: `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`
  - Weights: `indexer.k_norm`, `indexer.wk`, `indexer.wq_b`, `indexer.weights_proj`
- **SqrtSoftplus scoring** (V3 uses Sigmoid): `scoring_func="sqrtsoftplus"`
- **Clamped SwiGLU**: `swiglu_limit=10.0`, `gate.clamp(max=limit)`, `up.clamp(±limit)`
- **Grouped output LoRA**: `o_lora_rank=1024`, `o_groups=8` (V3 has no O-LoRA)
- **Hash-MoE bootstrap**: first 3 layers use static `tid2eid[input_id]` routing
- **No `kv_lora_rank`** (replaced by compress_ratios)
- **`head_dim=512`** (V3 uses `qk_nope_head_dim + qk_rope_head_dim`)
- **Separate Q/K/V projections**: `wq_a/wq_b`, `wk_a/wk_b`, `wv_a/wv_b` (V3 fuses to `fused_qkv_a_proj`)
- **Fused QKV alternative**: `wkv.weight` for some layers

Weight loading error showed 600+ uninitialized weights:
```
ValueError: Following weights were not initialized from checkpoint:
{'model.layers.X.self_attn.indexer.k_norm.weight',
 'model.layers.X.self_attn.fused_qkv_a_proj.weight',
 'model.layers.X.mlp.experts.w2_weight',
 'model.layers.X.mlp.shared_experts.gate_up_proj.weight',
 ...600+ more}
```

## Container's vllm Status

- Version: **vllm 0.11.0** (from `vLLM API server version 0.11.0`)
- `find vllm -name "*deepseek_v4*"` → **no results**
- `vllm/models/` directory doesn't exist (V4 uses new layout `vllm/models/deepseek_v4/`)
- **V4 support was added to upstream vllm after 0.11.0** (blog post 2026-04-24)

## Upstream V4 Code Available

- **HF transformers v5.8.0+**: `DeepseekV4ForCausalLM` (PR #45643, merged 2026-05-02)
- **Upstream vllm** (latest): `vllm/models/deepseek_v4/{nvidia,amd,xpu}/model.py`
  - XPU path exists: `vllm/models/deepseek_v4/xpu/model.py` (1298 lines)
  - Plus `sparse_mla.py` attention backend
  - Plus `_make_deepseek_v4_weights_mapper` for HF → vllm weight renaming
- **SGLang**: `python/sglang/srt/models/deepseek_v4_nextn.py`
- **PaddleFormers**: `paddleformers/transformers/deepseek_v4/modeling.py`
- **Tokenspeed**: `python/tokenspeed/runtime/models/deepseek_v4_mtp.py`

## Options Going Forward

### A. Cherry-pick upstream vllm V4 code into fork (~3000-5000 lines)
- Copy `vllm/models/deepseek_v4/` from upstream vllm
- Register `DeepseekV4ForCausalLM` properly
- Risk: might conflict with fork's XPU monkey patches
- Effort: several hours, high regression risk

### B. Use HF transformers directly (no vllm optimizations)
- Install transformers >= 5.8.0
- Use `AutoModelForCausalLM.from_pretrained()` with our BF16 weights
- Run inference in a simple loop
- Pro: Smallest code change
- Con: No TP, no PagedAttention, no continuous batching → very slow, single-device only

### C. Upgrade container's vllm to a version with V4
- Risk: breaks fork's XPU-specific patches (`vllm_kunlun`)
- Need to re-apply all monkey patches

### D. Accept limitation
- Conversion succeeded (543GB BF16 ready)
- Can't serve on this fork without major modeling code work
- Use V4 via HF transformers + custom inference loop (slow)

## Files

- `/home/vLLM-Kunlun-Aoripus/scripts/v4-flash-fp8-to-bf16.py` — conversion script (hand-parser)
- `/home/vLLM-Kunlun-Aoripus/vllm_kunlun/config/model.py` — whitelist (deepseek_v4 added)
- `/workspace/models/DeepSeek-V4-Flash-BF16/` — 543GB BF16 output, 46 shards
- `/tmp/v4_convert.log` — conversion log (success)
- `/tmp/v4_serve.log` — serve log (weight loading failure)
