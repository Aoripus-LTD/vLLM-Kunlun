# V4 Flash: Conversion & Fork Integration

## Status: CONVERSION ✅ | SERVE ❌ (architectural incompatibility + renamer impractical)

## What Worked

### 1. FP8/FP4 → BF16 Conversion (commits 7ea56b0, a26e878)
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

### 4. V4→V3 Weight Renamer (commit 624f3a6)
- mmap-based, streaming, zero-copy tensor views
- Maps V4 keys → V3 keys (simple renames + fusions)
- Strips V4-specific (hc_*, attn_sink, wkv, attn_norm, indexer.*, tid2eid)
- **NOT PRACTICAL**: shard 2+ takes 15+ min each (256 expert fusions/shard)
  - Estimated total: 10+ hours for 46 shards
  - Python GC + torch.cat overhead on 256 expert fusions/shard
  - Even with mmap (zero-copy reads), the fusion writes are slow

## What's Blocked

**V4 architecture is fundamentally incompatible with fork's V3 code.**

V4 introduces (per DeepSeek V4 paper + HF transformers v5.8.0 docs):
- **Manifold-Constrained Hyper-Connections (mHC)**: `hc_mult=4` parallel residual streams
- **Compressed Sparse Attention (CSA)**: compress KV 4x + Lightning Indexer
- **Heavily Compressed Attention (HCA)**: compress KV 128x, dense on compressed
- **Lightning Indexer**: `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`
- **SqrtSoftplus scoring** (V3 uses Sigmoid): `scoring_func="sqrtsoftplus"`
- **Clamped SwiGLU**: `swiglu_limit=10.0`
- **Grouped output LoRA**: `o_lora_rank=1024`, `o_groups=8`
- **Hash-MoE bootstrap**: first 3 layers use static `tid2eid[input_id]` routing
- **No `kv_lora_rank`** (replaced by compress_ratios)
- **`head_dim=512`** (V3 uses `qk_nope_head_dim + qk_rope_head_dim`)
- **Separate Q/K/V projections**: V3 fuses to `fused_qkv_a_proj`

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
- **V4 support was added to upstream vllm after 0.11.0** (blog post 2026-04-24)

## Upstream V4 Code Available

- **HF transformers v5.8.0+**: `DeepseekV4ForCausalLM` (PR #45643, merged 2026-05-02)
- **Upstream vllm** (latest): `vllm/models/deepseek_v4/{nvidia,amd,xpu}/model.py`
  - XPU path exists: `vllm/models/deepseek_v4/xpu/model.py` (1298 lines)
- **SGLang**: `python/sglang/srt/models/deepseek_v4_nextn.py`
- **Tokenspeed**: `python/tokenspeed/runtime/models/deepseek_v4_mtp.py`

## Paths Forward (require user decision)

### A. Cherry-pick upstream vllm V4 code into fork (~3000-5000 lines)
- Copy `vllm/models/deepseek_v4/` from upstream vllm
- Register `DeepseekV4ForCausalLM` properly
- Risk: might conflict with fork's XPU monkey patches
- Effort: several hours, high regression risk

### B. Use HF transformers directly (no vllm optimizations)
- Need transformers >= 5.8.0 (can't install in container: no pip, uv pip install no-ops)
- No TP, no PagedAttention, no continuous batching → very slow, single-device only

### C. Upgrade container's vllm to a version with V4
- Risk: breaks fork's XPU-specific patches (`vllm_kunlun`)
- Need to re-apply all monkey patches

### D. Accept limitation (current state)
- Conversion succeeded (543GB BF16 ready)
- Renamer written but too slow (10+ hours)
- Can't serve on this fork without major modeling code work
- Use V4 via HF transformers + custom inference loop (slow, different container)

## Commits This Session (pushed to releases/v0.11.0)

- `a26e878` — fix(v4-convert): hand-parse safetensors to bypass get_data absence
- `f44dddb` — feat(v4-flash): add deepseek_v4 to is_deepseek_mla whitelist
- `f306c15` — docs(v4-flash): document conversion success + serve incompatibility
- `624f3a6` — feat(v4-flash): add V4->V3 weight renamer (mmap-based, streaming)

## Files

- `/home/vLLM-Kunlun-Aoripus/scripts/v4-flash-fp8-to-bf16.py` — conversion script (hand-parser)
- `/home/vLLM-Kunlun-Aoripus/scripts/v4-flash-rename-for-v3.py` — V4→V3 renamer (mmap-based)
- `/home/vLLM-Kunlun-Aoripus/vllm_kunlun/config/model.py` — whitelist (deepseek_v4 added)
- `/workspace/models/DeepSeek-V4-Flash-BF16/` — 543GB BF16 output, 46 shards (V4 naming)
- `/workspace/models/DeepSeek-V4-Flash-V3compat/` — incomplete rename output (1 shard only)
- `/tmp/v4_convert.log` — conversion log (success)
- `/tmp/v4_serve.log` — serve log (weight loading failure)
- `/tmp/v4_rename.log` — rename log (stuck on shard 2)
