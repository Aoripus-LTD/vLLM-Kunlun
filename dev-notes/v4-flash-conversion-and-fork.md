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
2. → 读 convert_weight.py 在 V4 上的 dry-run（dry 模式：不写文件，只 log 哪些 weight 会被 dequant 哪些会被 keep）
3. → 试路径 A：从 HF 下载 modeling_deepseek_v4.py
4. → 试 BF16 转换 + 用 V3 类 serve（如果路径 A 失败）
5. → 评估 mHC 缺失的影响

## 已 push commits（截至本文件）
- `ac7d572` docs: pivot to DeepSeek V4 Flash research
- `1ccfee8` docs: GLM-5.1 debug log
- `39379ed` build: container symlink script
- `06772a2` fix: chatglm in is_deepseek_mla whitelist
- `46e6619` init: Aoripus fork baseline
