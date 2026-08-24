#!/bin/bash
set -eo pipefail
source /etc/profile.d/modules.sh
module load cuda13.0/toolkit/13.0.2
set -u
source /lambdafs/projects/reasoning-engine/model_serving/services/llama-3-70b-instruct/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/lambdafs/public_artifacts/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --max-model-len 4096 \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port 30092
