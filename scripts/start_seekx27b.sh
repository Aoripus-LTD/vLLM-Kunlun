#!/bin/bash
set -e
echo "=== 1. 杀旧 vllm ==="
for p in $(ps aux | grep -E "vllm|api_server|EngineCore|Worker_TP" | grep -v grep | awk '{print $2}'); do
  kill -9 $p 2>/dev/null
done
sleep 8

echo "=== 2. 启动 vllm serve (Qwen3.6-35B-A3B) ==="
source /workspace/vLLM-Kunlun/setup_env.sh
cd /workspace
export XACC_ENABLE_XPU=1
export XPURT_DISPATCH_MODE=0
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SEEKX_API_KEY=${SEEKX_API_KEY:-sk-CHANGE-ME-BEFORE-DEPLOY}

setsid nohup python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8356 \
  --model /home/models/Qwen35-122B-A10B-AB \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --tensor-parallel-size 8 \
  --dtype float16 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --distributed-executor-backend mp \
  --served-model-name seekx-27b-a7b-active \
  --chat-template /home/models/Qwen35-122B-A10B-AB/chat_template_seekx.jinja \
  --api-key ${SEEKX_API_KEY} \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  > /tmp/seekx27b_serve.log 2>&1 < /dev/null &
disown
sleep 2
echo "PID: $(ps aux | grep api_server | grep -v grep | awk '{print $2}' | head -1)"
