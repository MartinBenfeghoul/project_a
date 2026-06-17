#!/usr/bin/env bash

set -euo pipefail

model_name="mistralai/Mistral-7B-Instruct-v0.3"

usage() {
    echo "Usage: $0 TASK_NAME [MODEL_NAME]"
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 1
fi

task_name="$1"
if [ "$#" -eq 2 ]; then
    model_name="$2"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

python3 lm_eval_script.py \
    --model_name "${model_name}" \
    --tasks "${task_name}" \
    --limit 1 \
    --k_cache_type baseline \
    --v_cache_type baseline \
    --dump_full_kv_dir "${repo_root}/cache/kv_dumps"
