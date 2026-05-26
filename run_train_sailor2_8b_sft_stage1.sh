#!/usr/bin/env bash
set -e

export REPO=/home/llm3/Megatron-Sailor2
export MODEL_PATH=/home/llm3/models/sail/Sailor2-8B
export HF_HOME=/localhome/tansang/cache
export HF_DATASETS_CACHE=/localhome/tansang/cache/datasets

current_directory=$(dirname "$(readlink -f "$0")")

MODEL_PATH=/home/llm3/models/sail/Sailor2-8B

LOAD_DIR=${current_directory}/model/sailor2_8b_megatron_tp4_pp1
SAVE_DIR=${current_directory}/model/sailor2_8b_sft_stage1_tp4_pp1
DATA_PREFIX=${current_directory}/dataset/sailor2_8b_sft_stage1_preprocessed/train

# Stage-1 target setup:
# - one epoch over the full dataset (set STAGE1_TRAIN_SAMPLES accordingly)
# - global batch size 4096
# - linear lr decay 7e-6 -> 7e-7 across the epoch
STAGE1_GLOBAL_BATCH_SIZE=${STAGE1_GLOBAL_BATCH_SIZE:-4096}
STAGE1_TRAIN_SAMPLES=${STAGE1_TRAIN_SAMPLES:-0}

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

if [[ "${STAGE1_TRAIN_SAMPLES}" -le 0 ]]; then
  echo "Please set STAGE1_TRAIN_SAMPLES to the total number of training samples for one epoch."
  echo "Example: STAGE1_TRAIN_SAMPLES=1200000 ./run_train_sailor2_8b_sft_stage1.sh"
  exit 1
fi

mkdir -p ${SAVE_DIR}

LOG_ARGS="\
  --log_interval 1 \
  --save_interval 500 \
  --eval_interval 100 \
  --eval_iters 10"

TRAIN_ARGS="\
  --train_samples ${STAGE1_TRAIN_SAMPLES} \
  --lr_decay_samples ${STAGE1_TRAIN_SAMPLES} \
  --lr_decay_style linear \
  --weight_decay 0.1 \
  --lr_warmup_fraction 0.01 \
  --lr 7e-6 \
  --min_lr 7e-7 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_eps 1e-5"

DISTRIBUTED_ARGS="\
  --nproc_per_node 4 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port ${MASTER_PORT}"

QWEN_ARGS="\
  --use_rms_norm \
  --glu_activation swiglu \
  --no_tie_embed_logits \
  --no_new_tokens \
  --layernorm_epsilon 1e-6 \
  --rope_theta 1e6 \
  --use_flash_attn \
  --bf16 \
  --seq_length 4096"

COMMON_ARGS="\
  --hidden_dropout 0.0 \
  --attention_dropout 0.0 \
  --no_bias_gelu_fusion \
  --no_bias_dropout_fusion \
  --no_query_key_layer_scaling \
  --attention_softmax_in_fp32"

MEGATRON_TRAINS=(
  1
  ${DATA_PREFIX}
)

MEGATRON_VALIDS=(
  1
  ${DATA_PREFIX}
)

DATA_ARGS="\
  --train_data_path ${MEGATRON_TRAINS[*]} \
  --valid_data_path ${MEGATRON_VALIDS[*]} \
  --test_data_path ${MEGATRON_VALIDS[*]}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun ${DISTRIBUTED_ARGS} finetune.py \
  --tensor_model_parallel_size 4 \
  --pipeline_model_parallel_size 1 \
  --load ${LOAD_DIR} \
  --save ${SAVE_DIR} \
  --tensorboard_dir ${SAVE_DIR}/logs \
  --model_name qwen \
  --tokenizer_type Qwen2Tokenizer \
  --vocab_file ${MODEL_PATH} \
  --micro_batch_size 1 \
  --global_batch_size ${STAGE1_GLOBAL_BATCH_SIZE} \
  --data_type instruction \
  --variable_seq_lengths \
  --sequence_parallel \
  --no_gradient_accumulation_fusion \
  --recompute_granularity selective \
  --use_checkpoint_args \
  ${COMMON_ARGS} \
  ${LOG_ARGS} \
  ${TRAIN_ARGS} \
  ${QWEN_ARGS} \
  ${DATA_ARGS}
