# GLM-5.1 vLLM-Kunlun 适配调试日志

> 内部 debug 笔记，记录适配 zai-org/GLM-5.1-FP8 到 Aoripus-LTD/vLLM-Kunlun fork 的过程。
> 安全版本（不含 PAT / IP / hostname），可公开。

## 目标

把 GLM-5.1（zai-org/GLM-5，z.ai/THUDM，arxiv 2602.15763，2026-02 发布，MIT 协议）
部署到 vllm-kunlun 容器的 8 张 XPU 卡上。

## 模型参数

- 744B 总参 / 40B 激活
- 78 主层 + 1 MTP 层（layer 78）
- 256 routed experts / 8 top-k / 1 shared
- MLA-256: q_lora_rank=2048, kv_lora_rank=512, qk_head_dim=256
- DSA indexer: index_topk=2048, index_n_heads=32, index_head_dim=128
- checkpoint: FP8 (block 128×128) 或 W8A8 INT8 dynamic
- 架构名: `GlmMoeDsaForCausalLM` / `model_type: chatglm` (FP8 checkpoint) 或 `glm_moe_dsa` (INT8 checkpoint)

## 修复历史

| commit | 类型 | 说明 |
|---|---|---|
| `46e6619` | init | fork 初始状态（DeepseekV4 占位 + kimi_k25 注释 + 8 卡 mod） |
| `06772a2` | fix | `is_deepseek_mla` 白名单加 `chatglm` —— GLM-5.1 checkpoint `model_type=chatglm` 但架构是 DeepSeek-V3-style MLA+DSA+MoE，否则选到 non-MLA `DeepseekV2Attention` 撞 topk_indices_buffer 断言 |
| `39379ed` | build | 幂等的容器 symlink 脚本：把预装的 `/opt/vllm_kunlun/.../vllm_kunlun` 替换成 symlink 指向 fork —— 容器默认装的是 snapshot，不会自动用 fork 的代码 |

## 调试序列

### 错误 0：基础设置
- git PAT 配置 + 测试 `gh-proxy` 不支持 POST（`405 Method Not Allowed`）
- 改用 SSH 跳板（HK 47.86.108.117）+ SOCKS5 proxy `localhost:1080`
- Aoripus fork 设成 SOCKS5 走 git push

### 错误 1：`model_type: chatglm` 不被 transformers 4.57.0 认识
- **RED**: `AutoConfig.from_pretrained('/workspace/models/GLM5.1')` 失败
- **修复**: vllm-kunlun 的 `_XPU_CONFIG_REGISTRY` 已经 map `chatglm → ChatGLMConfig`，所以 `vllm.ModelConfig` 走通
- 不算 bug，是测试路径用了 AutoConfig 而不是 vllm ModelConfig

### 错误 2：`Model architectures ['GlmMoeDsaForCausalLM'] are not supported`
- **RED**: `vllm.ModelConfig` 报 not supported
- **根因**: fork 的 `register_model()` 只在 `vllm.general_plugins` 入口触发，不在 `vllm.platform_plugins`
- **解决**: 实际 `vllm serve` 走 engine init → 触发 `load_general_plugins()` → 自动加载。离线测试用 `vllm.plugins.load_general_plugins()` 手动触发

### 错误 3：`AssertionError: topk_indices_buffer is not supported for DeepseekV2Attention`
- **RED**: Worker_TP7 报断言
- **根因**: `is_deepseek_mla` 白名单没含 `chatglm` → `use_mla=False` → `DeepseekV2DecoderLayer` 选 `DeepseekV2Attention`（非 MLA）→ 该类的 `__init__` 断言 `topk_indices_buffer is None`，但 `is_v32` 路径会分配非 None 的 buffer
- **修复**: `vllm_kunlun/config/model.py` 白名单加 `chatglm`（line 13 的 `kv_lora_rank is not None` guard 保证老 ChatGLM-6B 不会被误判）
- 见 commit `06772a2`

### 错误 4：parent (EngineCore) 死，worker 报 TCPStore `Connection reset by peer`
- **根因**（**真正的**）：容器用的 `/opt/vllm_kunlun/.../vllm_kunlun` 是**预装 snapshot**，不是 fork！commit 进 fork 的 chatglm fix 容器**根本没看到**
- **修复**: 把预装的 vllm_kunlun 替换成 symlink 指向 fork
  - 大小写陷阱：容器路径是 `/workspace/vLLM-Kunlun-Aoripus/`（小写 v），不是大写 V
  - 写幂等脚本 `scripts/setup-container-symlink.sh`
  - 见 commit `39379ed`

### 错误 5：`--load-format dummy` → `param.data.to(torch.float16)` 在 Kunlun XPU 炸
- **错误**: `RuntimeError: [NOT IMPLEMENTED]: error code= 4(at .../copy_kernel.cpp:414)` —— vllm 上游 dummy loader 对每个浮点 param 做 `.to(float16)`，Kunlun XPU 的 copy_kernel 不支持
- **dummy 是测试用，不走真实部署路径** → drop `--load-format dummy`，用真实权重

### 错误 6（**当前硬墙**）：FP8 block-wise Triton kernel 在 Kunlun XPU 不可用
- **触发**: 真实 FP8 权重加载（142 safetensors shard，100% Completed [02:03]）后，`process_weights_after_loading` 调 `apply_w8a8_block_fp8_linear` → `per_token_group_quant_fp8`（Triton CUDA kernel）→ `RuntimeError: Triton Error [CUDA]: CUDA_ERROR_NOT_SUPPORTED`
- **根因**: `vllm/model_executor/layers/quantization/utils/fp8_utils.py` 用 Triton CUDA 实现。fork 的 `vllm_kunlun/ops/quantization/` **没有任何 fp8 实现**（只有 awq / gptq / moe_wna16 / compressed_tensors）。fork 的 `vllm_kunlun/ops/deep_gemm.py` 有 fp8 引用但只针对 DSA indexer 的 paged FP8 KV cache（`fp8_fp4_paged_mqa_logits`），不是 weight dequant
- **MiniCPM-1B 不炸的原因**: BF16 权重不走 FP8 路径
- **影响范围**: 所有用 FP8 block-wise (128×128) 量化的 GLM-5 / DeepSeek-V3 类模型

## 下一步选项

| 选项 | 描述 | 工作量 | 推荐度 |
|---|---|---|---|
| A | 把 FP8 权重 dequant 成 BF16，再走与 MiniCPM-1B 同样的 BF16 部署路径 | 写 `dequant_fp8_to_bf16.py`，1.5TB 磁盘 + 30-60 min | **★ 推荐** |
| B | 给 fork 加 FP8 block-wise dequant 的 Kunlun XPU 实现 | 写 C++/Triton-XPU kernel + 集成 | 长期方案 |
| C | 试 W8A8 INT8（`GLM-5-w4a8/`） | 改 serve 命令 | 同一 Triton 路径，**几乎肯定同样炸** |
| D | MoeWNA16 (W8A16) 路径 | 改 `quantization=moe_wna16` | GLM-5.1 没有这个量化版，不适用 |

## 已 commit + push 的修复
- `39379ed` build(scripts): add idempotent container symlink setup for vllm-kunlun
- `06772a2` fix(config): add chatglm to is_deepseek_mla whitelist for GLM-5.1
- `46e6619` init: Aoripus fork baseline + custom mods (vLLM-Kunlun)

## 部署模板（成功案例 — XPU）
```bash
docker exec vllm-kunlun bash -c '
  export VLLM_USE_V1=1 XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 --model /workspace/models/minicpm \
    --dtype bfloat16 --gpu-memory-utilization 0.85 --enforce-eager \
    --tensor-parallel-size 8 --max-num-seqs 256 \
    --served-model-name MiniCPM5-1B
'
```
