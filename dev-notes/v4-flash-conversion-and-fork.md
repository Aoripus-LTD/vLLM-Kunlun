# DeepSeek V4 Flash: 转换 + Fork 适配研究

> 续 dev-notes/v4-flash-overview.md（架构总览），聚焦**实际部署路径**。

## V4 Flash index 实际数据（容器内 /workspace/ 路径读到）

| 项 | 值 |
|---|---|
| 容器可见路径 | `/workspace/models/DeepSeek-V4-Flash/`（host: `/home/models/...`） |
| 总 keys | 69187 |
| 总 shards | 46 |
| 总大小 | 159609485896 bytes ≈ 149 GB（host）/ 156 GB（index metadata） |
| Shard 大小 | 1GB / 3.5GB 交替（bin/shard 切分，V3.2 风格） |
| Scale tensors (要被 convert 删) | 34257 |
| HC/mHC keys | `layers.X.hc_{attn,ffn}_{base,fn,scale}` + `hc_head_{base,fn,scale}` |
| layer 43 (MTP) | **不在 keys 里**（MTP 可能是 layer 42 之后的独立命名，或 `shared_head.*`） |
| config.json | `architectures: ["DeepseekV3ForCausalLM"]`，`model_type: "deepseek_v4"` |

## V4 vs V3 weight 命名差异（关键）

| V3.2 | V4 Flash |
|---|---|
| `self_attn.q_a_proj.weight` | `attn.wq_a.weight`（**注意 `attn` 不是 `self_attn`**）|
| `self_attn.q_b_proj.weight` | `attn.wq_b.weight` |
| `self_attn.kv_a_proj_with_mqa.weight` | `attn.wk_a.weight` + `attn.wv_a.weight`（**Q/K/V 分离**）|
| `self_attn.kv_b_proj.weight` | `attn.wk_b.weight` + `attn.wv_b.weight` |
| `self_attn.o_proj.weight` | `attn.wo_a.weight` + `attn.wo.weight` + `attn.wo_b.weight`（**O 用 LoRA 拆 3 段**）|
| `self_attn.indexer.wq_b.weight` | `attn.indexer.wq_b.weight` |
| N/A | `attn.indexer.wk.weight` + `attn.indexer.weights_proj.weight`（V3 没有）|
| N/A | `layers.X.hc_*`（**mHC，V4 独有**）|
| `mlp.experts.X.gate_proj.weight` | 待确认（V4 路径是 `mlp.experts.X.*`？还是新路径？）|
| `shared_experts.gate_up_proj.weight` | 待确认 |

## convert_weight.py 对 V4 Flash 的兼容性

✅ **能用的部分**（generic）：
- `weight`/`scale` 命名匹配（V4 沿用 V3.2 风格：`layers.X.attn.wq_a.weight` + `layers.X.attn.wq_a.scale`）
- 顶层 `model.safetensors.index.json` 处理
- sharded safetensors 逐个读、dequant、写
- `is_expert_weight("experts" in name)` 需要 V4 沿用 `experts` 路径

❌ **可能炸的部分**：
- 文档说 "V3.2 / 62 layers / 163 shards"——V4 Flash 是 43 layers / 46 shards，脚本是 generic（按 index 遍历）应该 OK
- `dequant_fp4_weight` 用 `FP4_TABLE` 查表 + E8M0 scale——V4 expert 是 MXFP4，理论上兼容（MXFP4 标准）
- mHC tensors 是 BF16 不会触发 FP8/FP4 路径

⚠️ **需要验证的**：
- V4 expert 路径是否真的叫 `experts`（而不是新命名如 `experts_v4`）
- V4 的 E8M0 scale 存储格式是否跟 V3.2 完全一致
- `attn.wo_a`/`wo_b` 这种 O LoRA 拆分是否被脚本正确处理

## 部署路径决策

V4 Flash 有 `auto_map: {"AutoModelForCausalLM": "DeepseekV4ForCausalLM"}` 引用**自定义 model class**——但本地**没有 modeling_*.py**（HF repo 提供，下载时被剥离了）。

**3 个路径**：

### 路径 A：下载 HF 自定义 modeling code（推荐）
- 从 `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash` 拉 `modeling_deepseek_v4.py` + `configuration_deepseek_v4.py`
- 走 `trust_remote_code=True` 路径
- 自带 mHC + CSA+HCA + O LoRA + sqrtsoftplus + 1M context 全套
- **风险**：trust_remote_code 在 vllm 路径下需要特殊处理（fork 目前的代码可能不直接支持 trust_remote_code 自定义类）
- **需要**：HF 下载走 SOCKS proxy

### 路径 B：写 minimal V3-compatible 类
- 复用 fork 的 `DeepseekV2ForCausalLM`（已有），把 V4 weight name 映射到 V3 内部 naming
- 缺点：mHC / CSA+HCA / O LoRA 全丢——模型输出质量灾难
- **不推荐**

### 路径 C：写完整 V4 类
- 实现 mHC, CSA+HCA hybrid attention, O LoRA, sqrtsoftplus scoring, 1M context
- 大量工作（1-2 周）
- **长期方案**

## 优先级

1. ✅ 确认 V4 Flash 存在 + index 可读
2. ✅ **convert_weight.py dry-run on V4 Flash（关键发现）**：
   - Model index 完美解析：69187 keys / 33792 FP4 expert / 375 FP8 non-expert / 34167 scales
   - **第一 shard 加载就炸**：`AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`
   - **根因**：V4 用 `float8_e8m0fnu` (PyTorch 2.4+) 做 WKV fused QKV weight 的 scale。容器 torch 旧版本没这个 dtype attribute
   - **修复方向**：在 fork 写 V4-specific 转换脚本（不用 dtype attribute，按 E8M0 字节 raw 解析）
3. → 试 BF16 转换 + V3 类 serve
4. → 试路径 A：HF 下载 modeling code
5. → 评估 mHC 缺失的影响

## V4 Flash weight 命名（dry-run 实际发现）

| V3.2 | V4 Flash (实测) |
|---|---|
| `self_attn.q_a_proj.weight` | `attn.wq_a.weight` |
| `self_attn.q_b_proj.weight` | `attn.wq_b.weight` |
| `self_attn.kv_a_proj_with_mqa.weight` | `attn.wk_a.weight` + `attn.wv_a.weight`（Q/K/V 分离）|
| `self_attn.kv_b_proj.weight` | `attn.wk_b.weight` + `attn.wv_b.weight` |
| `self_attn.o_proj.weight` | `attn.wo_a.weight` + `attn.wo_b.weight`（**O LoRA**）|
| N/A | `attn.wkv.weight`（**fused QKV，FP8 e8m0fnu**）|
| N/A | `attn.attn_sink`（**V4 新增**）|
| N/A | `attn.kv_norm.weight` + `attn.q_norm.weight`（**per-head RMSNorm**）|
| N/A | `attn_norm.weight`（**post-attn norm**）|
| `self_attn.indexer.wq_b` | `attn.indexer.wq_b`（DSA 一样）|
| `mlp.experts.X.gate_proj.weight` | `ffn.experts.X.w1.weight`（**w1/w2/w3 替代 gate/up/down**）|
| `shared_experts.gate_up_proj.weight` | 待确认 |
| N/A | `layers.X.hc_attn_base/fn/scale` + `hc_ffn_base/fn/scale`（**mHC**）|
| N/A | `hc_head_base/fn/scale`（**head-level mHC**）|

**layer 0 完整 key 列表（实测，1565 keys/层）**：
```
layers.0.attn.attn_sink                              ← V4 新增（attention sink）
layers.0.attn.kv_norm.weight                         ← V4 新增（KV norm）
layers.0.attn.q_norm.weight                          ← V4 新增（Q norm）
layers.0.attn.wkv.scale / wkv.weight                 ← V4 fused QKV (FP8 e8m0fnu scale)
layers.0.attn.wo_a.scale / wo_a.weight               ← O LoRA down
layers.0.attn.wo_b.scale / wo_b.weight               ← O LoRA up
layers.0.attn.wq_a.scale / wq_a.weight               ← Q LoRA down
layers.0.attn.wq_b.scale / wq_b.weight               ← Q LoRA up
layers.0.attn.wk_a.scale / wk_a.weight               ← K LoRA down (V4 单独)
layers.0.attn.wk_b.scale / wk_b.weight               ← K LoRA up
layers.0.attn.wv_a.scale / wv_a.weight               ← V LoRA down
layers.0.attn.wv_b.scale / wv_b.weight               ← V LoRA up
layers.0.attn_norm.weight                            ← post-attention RMSNorm
layers.0.ffn.experts.0.w1/w2/w3.{weight,scale}       ← 256 个 expert × 3 weights (FP4 MXFP4)
layers.0.hc_attn_base/fn/scale                        ← mHC (V4 独有)
layers.0.hc_ffn_base/fn/scale                         ← mHC
```

**专家权重的 w1/w2/w3 vs gate/up/down 命名约定**：
- `w1` = gate_proj（input → intermediate）
- `w2` = down_proj（intermediate → output，in MoE gate context）—— 注意：这是 **post-MoE-gate 的 down_proj**，不是普通 MLP 的 down_proj
- `w3` = up_proj（input → intermediate）

## 容器 torch 版本 vs V4 Flash 需求

| dtype | 容器支持 | V4 用法 |
|---|---|---|
| `torch.float8_e4m3fn` | ✅（vllm_kunlun torch 2.0+）| 部分 FP8 权重（attn 一些） |
| `torch.float8_e8m0fnu` | ❌（需要 torch 2.4+，容器是更早版本）| **WKV fused QKV 的 scale**（E8M0）|
| `torch.bfloat16` | ✅ | norm, bias, gate, hc_* 等 |
| `torch.float32` | ✅ | indexer.k_norm 等 |
| FP4 (E2M1 packed in int8) | 需手动解析 | `ffn.experts.X.w1/w2/w3.weight` |

**结论**：必须用 byte-level E8M0 解析，不能依赖 dtype attribute。

## 已 push commits（截至本文件）
- `ac7d572` docs: pivot to DeepSeek V4 Flash research
- `1ccfee8` docs: GLM-5.1 debug log
- `39379ed` build: container symlink script
- `06772a2` fix: chatglm in is_deepseek_mla whitelist
- `46e6619` init: Aoripus fork baseline
