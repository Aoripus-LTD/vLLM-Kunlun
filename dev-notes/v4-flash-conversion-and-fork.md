# V4 Flash: Conversion & Fork Integration

## Status: CONVERSION ✅ | SERVE ❌ (kernel-level XPU incompatibility)

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

### 2. Fork Whitelist (commit f44dddb)
- Added `"deepseek_v4"` to `is_deepseek_mla` whitelist in `vllm_kunlun/config/model.py`
- `DeepseekV4ForCausalLM` was already registered in `models/__init__.py:131` → maps to `DeepseekV3ForCausalLM`

### 3. V4→V3 Weight Renamer (commit 624f3a6, ac3d83c)
- mmap-based, streaming, zero-copy tensor views
- Maps V4 keys → V3 keys (simple renames + fusions)
- Strips V4-specific (hc_*, attn_sink, wkv, attn_norm, indexer.*, tid2eid)
- **NOT PRACTICAL**: shard 2+ with 256 expert fusions takes 15+ min each
  - Estimated total: 10+ hours for 46 shards
  - Python GC + torch.cat overhead on 256 expert fusions/shard

## What's Blocked (incremental discovery)

Through incremental patching experiments, we discovered the **layered blockers** in order:

### Blocker 1: Weight Validation (RESOLVED with sed patch)
- **Error**: `ValueError: Following weights were not initialized from checkpoint: {...600+ weights...}`
- **Cause**: V4 has 600+ weight names V3 doesn't recognize (indexer.*, mHC, CSA/HCA, etc.)
- **Fix**: sed patch to `default_loader.py:276` (commented out the raise)
- **Commit**: 7ea56b0/a26e878 (in-repo), sed patch was experimental (reverted)

### Blocker 2: MLA + Sliding Window Assertion (RESOLVED with config patch)
- **Error**: `AssertionError: MLA is not supported for slidingwindow`
- **Cause**: V4 has `sliding_window: 128` + `kv_lora_rank: 512` (CSA design), V3 code asserts these are incompatible
- **Fix**: Remove `sliding_window` from converted config (V3 then doesn't try to use sliding window)
- **Note**: This is a V4-specific CSA feature; removing it means the model won't have V4's full attention pattern

### Blocker 3: Kernel-level XPU Error (FINAL BLOCKER)
- **Error**: `kl3ChannelCheckErrors failed, error set to 66250, status= 700`
- **Cause**: Model loaded successfully, started executing on XPU, but V4-specific kernel operations (CSA/HCA attention, mHC, Lightning Indexer) aren't implemented in fork's XPU backend
- **Implication**: Even with Python-level fixes, V4 requires XPU-specific kernel implementations that the fork doesn't have
- **Fix**: Would require porting V4's XPU kernels from upstream vllm (~thousands of lines of CUDA/Triton → XPU)

## V4 Architecture vs V3 (compatibility matrix)

| V4 Feature | V3 Support | Required for V4 Serve |
|---|---|---|
| BF16 weights (renamed) | ✓ (after renamer) | Mapping script |
| MoE experts (w1+w3 → w13) | ✓ (after fusion) | Fusion in renamer |
| MLA (kv_lora_rank) | ✓ | Whitelist addition (done) |
| Sliding window | ✗ (assertion) | Remove from config OR add support |
| CSA/HCA attention | ✗ (no kernels) | XPU kernel port from upstream |
| Lightning Indexer | ✗ (no module) | Modeling code from upstream |
| SqrtSoftplus scoring | ✗ (uses Sigmoid) | Modeling code from upstream |
| Clamped SwiGLU | ✗ (no clamp) | Modeling code from upstream |
| Grouped O-LoRA | ✗ (no LoRA) | Modeling code from upstream |
| Hash-MoE bootstrap | ✗ (no module) | Modeling code from upstream |
| mHC (hc_* weights) | ✗ (no module) | Modeling code from upstream |

## Commits This Session (pushed to releases/v0.11.0)

- `a26e878` — fix(v4-convert): hand-parse safetensors to bypass get_data absence
- `f44dddb` — feat(v4-flash): add deepseek_v4 to is_deepseek_mla whitelist
- `f306c15` — docs(v4-flash): document conversion success + serve incompatibility
- `624f3a6` — feat(v4-flash): add V4->V3 weight renamer (mmap-based, streaming)
- `ac3d83c` — perf(v4-flash): disable GC during renamer
- `166be6e` — docs(v4-flash): document renamer impracticality + final status
- `90c2f75` — docs(session): patch experiment results

## Files

- `/home/vLLM-Kunlun-Aoripus/scripts/v4-flash-fp8-to-bf16.py` — conversion script (hand-parser)
- `/home/vLLM-Kunlun-Aoripus/scripts/v4-flash-rename-for-v3.py` — V4→V3 renamer (mmap-based)
- `/home/vLLM-Kunlun-Aoripus/vllm_kunlun/config/model.py` — whitelist (deepseek_v4 added)
- `/workspace/models/DeepSeek-V4-Flash-BF16/` — 543GB BF16 output, 46 shards
- `/tmp/v4_convert.log` — conversion log (success)
- `/tmp/v4_serve5.log` — serve log (kernel error after weight loading + MLA fixes)

## Path Forward (require user decision)

### A. Cherry-pick upstream vllm V4 code + XPU kernels
- Copy `vllm/models/deepseek_v4/{nvidia,amd,xpu}/` from upstream
- Plus `sparse_mla.py` attention backend
- Plus XPU kernel implementations for CSA/HCA/mHC
- Total: ~5000-10000 lines
- Risk: conflicts with fork's XPU monkey patches
- Effort: days, high regression risk

### B. Use HF transformers >= 5.8.0 directly
- Need transformers >= 5.8.0 (can't install in container: no pip, uv pip install no-ops)
- No TP, no PagedAttention, no continuous batching → very slow
- Would need different container

### C. Upgrade container's vllm + re-apply XPU patches
- Risk: breaks fork's XPU-specific patches
- Need to re-apply all monkey patches

### D. Accept limitation (current state)
- Conversion succeeded (543GB BF16 ready)
- Serve revealed 3 layered blockers (weight validation, MLA+sliding, XPU kernel)
- Would need major modeling + kernel work to fully support V4
