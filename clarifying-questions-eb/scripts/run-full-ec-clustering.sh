#!/bin/bash
set -euo pipefail

# Run the frozen confirmatory E-C clustering analyses in one Slurm step. Keeping
# the server and both clients in this process avoids relying on a second SSH
# admission while a long-running allocation is close to its wall-time limit.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_COMMIT="${PROMPT_COMMIT:-79554bc}"
CLUSTERER_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
CLUSTERER_REVISION="50fd307e57011801c7833c87efa1984ddf2db42f"
PORT="${PORT:-30093}"
WORKERS="${WORKERS:-10}"
RESUME="${RESUME:-0}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
SERVER_LOG="${PROJECT_DIR}/scan_results/ec_clustering_full_server.log"

QWEN_SOURCE="scan_results/ec_qwen3_8b_results.jsonl"
QWEN_PILOT="scan_results/ec_clustering_pilot_final_qwen3_8b.jsonl"
QWEN_OUT="scan_results/ec_clustering_full_qwen3_8b.jsonl"
LLAMA_SOURCE="scan_results/ec_llama3_70b_results.jsonl"
LLAMA_PILOT="scan_results/ec_clustering_pilot_final_llama3_70b.jsonl"
LLAMA_OUT="scan_results/ec_clustering_full_llama3_70b.jsonl"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "error: run this script inside the allocated dgx-26 Slurm job" >&2
    exit 2
fi
if [[ "$(hostname -s)" != "dgx-26" ]]; then
    echo "error: expected dgx-26, got $(hostname -s)" >&2
    exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
    echo "error: RESUME must be 0 or 1" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
for path in "${QWEN_SOURCE}" "${QWEN_PILOT}" "${LLAMA_SOURCE}" "${LLAMA_PILOT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "error: required input is missing: ${path}" >&2
        exit 2
    fi
done
if [[ "${RESUME}" == "0" ]]; then
    for path in "${QWEN_OUT}" "${LLAMA_OUT}"; do
        if [[ -e "${path}" || -e "${path}.summary.json" ]]; then
            echo "error: output exists: ${path}; use RESUME=1 only for this exact frozen run" >&2
            exit 2
        fi
    done
fi

source /etc/profile.d/modules.sh
module load cuda13.0/toolkit/13.0.2
source /lambdafs/projects/reasoning-engine/model_serving/services/llama-3-70b-instruct/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/lambdafs/public_artifacts/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve "${CLUSTERER_MODEL}" \
    --revision "${CLUSTERER_REVISION}" \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

stop_server() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap stop_server EXIT INT TERM

echo "waiting for ${CLUSTERER_MODEL} on ${BASE_URL} (pid=${SERVER_PID})"
for _ in $(seq 1 180); do
    if curl --fail --silent "${BASE_URL}/models" >/dev/null; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "error: vLLM exited before readiness; see ${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 5
done
if ! curl --fail --silent "${BASE_URL}/models" >/dev/null; then
    echo "error: vLLM was not ready after 15 minutes; see ${SERVER_LOG}" >&2
    exit 1
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
    RESUME_ARGS+=(--resume)
fi

run_subject() {
    local source=$1
    local output=$2
    local pilot=$3
    local subject_model=$4
    echo "starting ${subject_model}: ${source} -> ${output}"
    python -m ec.run_clustering \
        --source "${source}" \
        --out "${output}" \
        --base-url "${BASE_URL}" \
        --clusterer-model "${CLUSTERER_MODEL}" \
        --clusterer-revision "${CLUSTERER_REVISION}" \
        --subject-model "${subject_model}" \
        --prompt-commit "${PROMPT_COMMIT}" \
        --exclude-ids-from "${pilot}" \
        --workers "${WORKERS}" \
        --bootstrap-samples 10000 \
        --bootstrap-seed 0 \
        --confidence-level 0.95 \
        "${RESUME_ARGS[@]}"
}

# Deliberately sequential: the same frozen clusterer serves both saved
# subject-model datasets, and the Qwen run must finish before Llama starts.
run_subject "${QWEN_SOURCE}" "${QWEN_OUT}" "${QWEN_PILOT}" "Qwen/Qwen3-8B"
run_subject "${LLAMA_SOURCE}" "${LLAMA_OUT}" "${LLAMA_PILOT}" "meta-llama/Meta-Llama-3-70B-Instruct"

echo "full E-C clustering runs completed"
