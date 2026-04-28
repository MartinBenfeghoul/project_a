#!/bin/bash
#SBATCH --job-name=gist_vs_details
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00

set -e

# -----------------------------
# Default configuration
# -----------------------------
#MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="./results"
TASKS="ruler"
#TASKS="longbench"
LIMIT="50"
LOG_EFFICIENCY="true"
DEBUG="false"

# Key cache defaults
K_CACHE_TYPE="baseline"
DECOMPOSITION_METHOD="svd"
COMP_RATIO="2.0"
ENERGY_THRESHOLD="0.95"
RANK_SELECTION="comp_ratio"
K_LR="1e-2"
N_ITER="3"

# Value cache defaults
V_CACHE_TYPE="mlp"
NUM_LAYERS_PER_MLP="$(printf '2 %.0s' {1..16})$(printf '2 %.0s' {1..16})"
HIDDEN_FACTORS_PER_MLP="$(printf '4 %.0s' {1..32})"
NUM_HEADS_PER_MLP="$(printf '8 %.0s' {1..16})$(printf '8 %.0s' {1..16})"
PER_SEQUENCE="true"
TARGET_PERC="$(printf '75 %.0s' {1..16})$(printf '75 %.0s' {1..16})"
TARGET_MODEL_NUM_HEADS="8"
V_LR="1e-3"
OPTIMIZER="adam"
LOSS_FUNC="mse"
#LOSS_FUNC="huber_loss"
NUM_EPOCHS="400"
META_WEIGHTS_PATH=""
NORMALIZE_VALUES=false
UN_ROPE=true
GLOBAL_COMPRESSION=false
MAX_ERROR=false
ADD_LINEAR=false
TARGET_CR="4.0"
# -----------------------------
# Flags
# -----------------------------
EFF_FLAG=""
DEBUG_FLAG=""
PER_SEQ_FLAG=""
NORM_VAL_FLAG=""
UN_ROPE_FLAG=""
GLOB_COMPR_FLAG=""
MAX_ERROR_FLAG=""
ADD_LINEAR_FLAG=""

if [ "$LOG_EFFICIENCY" = "true" ]; then
  EFF_FLAG="--log_efficiency_metrics"
fi

if [ "$DEBUG" = "true" ]; then
  DEBUG_FLAG="--debug"
fi

if [ "$PER_SEQUENCE" = "true" ]; then
  PER_SEQ_FLAG="--per_sequence"
fi

if [ "$NORMALIZE_VALUES" = "true" ]; then
  NORM_VAL_FLAG="--normalize_values"
fi

if [ "$UN_ROPE" = "true" ]; then
  UN_ROPE_FLAG="--un_rope"
fi

if [ "$GLOBAL_COMPRESSION" = "true" ]; then
  GLOB_COMPR_FLAG="--global_compression"
fi

if [ "$MAX_ERROR" = "true" ]; then
  MAX_ERROR_FLAG="--max_error"
fi

if [ "$ADD_LINEAR" = "true" ]; then
  ADD_LINEAR_FLAG="--add_linear"
fi


# -----------------------------
# Run evaluation
# -----------------------------

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate gist_vs_details

python value_test.py \
  --model_name $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --tasks $TASKS \
  $EFF_FLAG \
  $DEBUG_FLAG \
  $PER_SEQ_FLAG \
  $NORM_VAL_FLAG \
  $UN_ROPE_FLAG \
  $GLOB_COMPR_FLAG \
  $MAX_ERROR_FLAG \
  $ADD_LINEAR_FLAG \
  -kc $K_CACHE_TYPE \
  --decomposition_method $DECOMPOSITION_METHOD \
  -r $COMP_RATIO \
  -e $ENERGY_THRESHOLD \
  --rank_selection $RANK_SELECTION \
  --k_lr $K_LR \
  --n_iter $N_ITER \
  -vc $V_CACHE_TYPE \
  --num_layers_per_mlp $NUM_LAYERS_PER_MLP \
  --hidden_factors_per_mlp $HIDDEN_FACTORS_PER_MLP \
  --num_heads_per_mlp $NUM_HEADS_PER_MLP \
  --target_perc $TARGET_PERC \
  --target_model_num_heads $TARGET_MODEL_NUM_HEADS \
  --v_lr $V_LR \
  --optimizer $OPTIMIZER \
  --loss_func $LOSS_FUNC \
  --num_epochs $NUM_EPOCHS \
  --limit $LIMIT \
  --target_cr $TARGET_CR \
  $META_WEIGHTS_PATH
