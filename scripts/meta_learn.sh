#!/bin/bash --login
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:1
#SBATCH --job-name=slurm
#SBATCH --time=5-00:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

NTFY_TOPIC="cluster_notifications_teresadelgado"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

OUT_LOG="${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-noid}.out"

send_ntfy () {
  title="$1"
  body="$2"
  curl -fsS \
    -H "Title: ${title}" \
    -d "$body" \
    "$NTFY_URL" >/dev/null || true
}

on_exit () {
  rc=$?
  if [ $rc -eq 0 ]; then
    send_ntfy "SLURM DONE" "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) finished successfully."
  else
    err_tail=""
    if [ -s "$OUT_LOG" ]; then
      err_tail="$(tail -n 80 "$OUT_LOG")"
    else
      err_tail="(No log output found yet.)"
    fi

    send_ntfy "SLURM FAILED (rc=$rc)" \
"Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) failed on ${SLURMD_NODENAME:-unknown}

Last log lines:
$err_tail"
  fi
}
trap on_exit EXIT



base_dir="/home/t84411738/github/gist_vs_details"

cd "${base_dir}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pem-llm
  
python3 meta_learning.py
