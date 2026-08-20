#!/usr/bin/env bash
# Serve Qwen2.5-3B-Instruct for Jac-Rag-GPT-vllm on :8001.
# Tool calling (search_docs) needs the hermes parser; prefix caching is on by
# default in V1 and the profiler measures its effect.
set -euo pipefail

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 2000 ]; then
    echo "GPU already has ${USED} MiB in use — check for leaked VLLM::EngineCore:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    exit 1
fi

source /home/xiaoyu/miniconda3/etc/profile.d/conda.sh
conda activate spork

exec vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --gpu-memory-utilization 0.75 \
    --max-model-len 16384
