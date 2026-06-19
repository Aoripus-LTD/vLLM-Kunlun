# DeepSeek V4 Flash vLLM-Kunlun 适配

> 续 GLM-5.1 调试（命中 FP8 Triton 硬墙），改做 DeepSeek V4 Flash。
> DeepSeek V4 Pro（1.6T/49B）未本地下载，V4 Flash（284B/13B）可用。

## 目标

把 DeepSeek V4 Flash 部署到 vllm-kunlun 8 张 XPU 卡上。

## V4 家族架构（来自 README + config.json）

| 维度 | V4 Flash | V4 Pro | GLM-5.1（对比）|
|---|---|---|---|
| 总参 | 284B | 1.6T | 744B |
| 激活 | 13B | 49B | 40B |
| 上下文 | 1M | 1M | 202K |
| 层数 | 43 | ? | 78+1 MTP |
| Hidden | 4096 | ? | 6144 |
| 精度 | FP4+FP8 mixed | FP4+FP8 mixed | FP8 / INT8 |
| HF 链接 | deepseek-ai/DeepSeek-V4-Flash | deepseek-ai/DeepSeek-V4-Pro | zai-org/GLM-5 |
| ModelScope | deepseek-ai/DeepSeek-V4-Flash | deepseek-ai/DeepSeek-V4-Pro | ZhipuAI/GLM-5 |

## V4 架构新特性（README §Introduction）

1. **Hybrid Attention Architecture: CSA + HCA**
   - **CSA (Compressed Sparse Attention)** + **HCA (Heavily Compressed Attention)** —— 组合使用，大幅提升长上下文效率
   - V4 Pro 在 1M context 下只用 V3.2 的 **27% 单 token 推理 FLOPs** + **10% KV cache**
2. **mHC (Manifold-Constrained Hyper-Connections)** —— 替代传统残差连接，增强信号传播稳定性
3. **Muon Optimizer** —— 训练更快更稳
4. **预训练 32T tokens**（vs GLM-5 28.5T）
5. **1M context** with `yarn` rope scaling + `compress_rope_theta: 160000` + `compress_ratios` per layer + `sliding_window: 128`

## V4 Flash config.json 关键字段

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],    // 复用 V3 架构名
  "model_type": "deepseek_v4",                  // ★ 新的 model_type（fork 没认）
  "expert_dtype": "fp4",                        // ★ 专家用 FP4 MXFP4 E2M1
  "quantization_config": {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",                      // V4 用 E8M0 scale（V3 用 float8_e8m0）
    "weight_block_size": [128, 128]
  },
  "hidden_size": 4096,
  "num_hidden_layers": 43,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,                     // MLA 风格
  "q_lora_rank": 1024,
  "o_lora_rank": 1024,                          // ★ O 也用 LoRA 压缩
  "qk_rope_head_dim": 64,
  "head_dim": 512,
  "max_position_embeddings": 1048576,          // 1M
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 2048,
  "index_topk": 512,                            // DSA indexer
  "index_n_heads": 64,
  "index_head_dim": 128,
  "hc_eps": 1e-06, "hc_mult": 4, "hc_sinkhorn_iters": 20,   // ★ mHC
  "o_groups": 8,                               // ★ O projection 分组
  "scoring_func": "sqrtsoftplus",              // ★ V4 新 scoring
  "routed_scaling_factor": 1.5,
  "topk_method": "noaux_tc",
  "num_nextn_predict_layers": 1,               // 1 MTP layer
  "sliding_window": 128, "swiglu_limit": 10.0,
  "compress_rope_theta": 160000, "compress_ratios": [...],   // ★ per-layer 压缩
  "auto_map": {                                 // ★ 自定义 model code
    "AutoModelForCausalLM": "DeepseekV4ForCausalLM",
    "AutoModel": "DeepseekV4ForCausalLM"
  }
}
```

## Aoripus 的部署策略（关键发现）

`/home/models/Deepseek-V4-Flash-Aoripus/convert_weight.py` 的 docstring 写着：

> **DeepSeek-V3.2 FP8/FP4 -> BF16 Converter**
> Key differences from V3:
>   - Scale format: ue8m0 (unsigned E8M0, power-of-2 scales, may be stored as uint8)
>   - weight_dequant uses block-reshape approach (from V3.2 model.py:490)
>   - DSA (DeepSeek Attention): each layer has `indexer` submodule
>       - indexer.wq_b, indexer.wk: FP8 (have weight_scale_inv)
>       - indexer.k_norm (weight+bias), indexer.weights_proj: BF16 (no scale)
>   - MTP layer (layer 61): enorm, hnorm, eh_proj, shared_head.norm/head, embed_tokens
>   - No dependency on kernel.py (pure PyTorch)
>   - 62 layers (0-61), 163 shards, ~92k keys
>   - Experts weights use FP4 (MXFP4 E2M1, packed 2 per uint8, group_size=32, E8M0 scale)

**这意味着 Aoripus 的部署策略是：把 FP8/FP4 权重 dequant 到 BF16，再走 BF16 serve 路径**——与 MiniCPM-1B 部署完全一致，**完美绕过 GLM-5 撞的 Triton FP8 硬墙**。

V4 Flash 149GB FP4+FP8 mixed → dequant → ~570GB BF16 → serve。

## 进度

- [x] 4 个 GLM-5 commits push（chatglm fix、symlink 脚本、dev-notes）
- [x] 用户确认切换：FP8 Triton 墙 → V4 Flash，Aoripus 转换策略避墙
- [ ] run `convert_weight.py` 把 V4 Flash 转换成 BF16（输出到 `/home/models/DeepSeek-V4-Flash-BF16/`）
- [ ] fork 适配 `model_type: "deepseek_v4"`（加 `is_deepseek_mla` 白名单）
- [ ] 处理 `auto_map` 自定义 model code（`DeepseekV4ForCausalLM`）—— 需要在 fork 里有对应类
- [ ] 移除 `--load-format dummy`，BF16 serve 真实权重
- [ ] curl 推理验证
- [ ] 回归 MiniCPM-1B

## 已知风险

1. **auto_map 类的实现**：`DeepseekV4ForCausalLM` 可能在 `DeepSeek-V4-Flash/` 仓库的 `modeling_*.py` 里（`trust_remote_code=True` 加载）。fork 需要 `from vllm.model_executor.models import DeepseekV3ForCausalLM` 注册同名类——可复用现有 `GlmMoeDsaForCausalLM` 模式（一个一行继承）
2. **mHC (Manifold-Constrained Hyper-Connections)**：需要在 `DeepseekV2DecoderLayer` 的 forward 里支持 `hc_mult=4` 个并行流 + `hc_sinkhorn_iters=20` 双随机矩阵投影
3. **CSA+HCA hybrid attention**：每层可能在 CSA 和 HCA 间交替，需要看 layer config 决定
4. **FP4 专家**（如果走 dequant 路径就不需要）—— convert_weight.py 已经处理
5. **O LoRA + O groups**：`o_lora_rank=1024` 和 `o_groups=8` 是 V4 新设计，需要 MLA 类支持
6. **compress_rope_theta + compress_ratios**：per-layer 压缩率，per-layer RoPE 调整
7. **1M context**：`max_position_embeddings=1048576` 比 V3 4x，需要 RoPE 扩展和 KV cache 容量检查
8. **scoring_func="sqrtsoftplus"**：V3 sigmoid → V4 sqrtsoftplus，MoE gate 计算差异
