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
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port 30093
