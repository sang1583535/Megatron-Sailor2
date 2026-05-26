#!/usr/bin/env bash
set -e

current_directory=$(dirname "$(readlink -f "$0")")

MODEL_PATH=/home/llm3/models/sail/Sailor2-8B

HF_TO_MEGATRON_DIR=${current_directory}/model/sailor2_8b_megatron
TP4_DIR=${current_directory}/model/sailor2_8b_megatron_tp4_pp1

mkdir -p ${HF_TO_MEGATRON_DIR}
mkdir -p ${TP4_DIR}

python weights_conversion/hf_to_megatron.py qwen \
  --size 8 \
  --out ${HF_TO_MEGATRON_DIR} \
  --model-path ${MODEL_PATH}

python tools/checkpoint_util.py \
  --target_tensor_parallel_size 4 \
  --target_pipeline_parallel_size 1 \
  --load_dir ${HF_TO_MEGATRON_DIR} \
  --save_dir ${TP4_DIR} \
  --model_type qwen \
  --true_vocab_size 151936 \
  --megatron_path ${current_directory} \
  --vocab_file ${MODEL_PATH} \
  --bf16
