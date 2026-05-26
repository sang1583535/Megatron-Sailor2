#!/usr/bin/env bash
set -e

source ~/miniconda3/bin/activate
conda activate megatron_sailor2

current_directory=$(dirname "$(readlink -f "$0")")

MODEL_PATH=/home/llm3/models/sail/Sailor2-8B
INPUT_JSONL=${current_directory}/dataset/sailor2_sft_stage1_jsonl/train.jsonl
OUTPUT_PREFIX=${current_directory}/dataset/sailor2_8b_sft_stage1_preprocessed/train

mkdir -p "$(dirname ${OUTPUT_PREFIX})"

python tools/preprocess_instruct_data.py \
  --input ${INPUT_JSONL} \
  --output_prefix ${OUTPUT_PREFIX} \
  --tokenizer_type Qwen2ChatTokenizer \
  --vocab_file ${MODEL_PATH} \
  --chunk_size 32 \
  --workers 64 \
  --log_interval 20000 \
  --question_key user \
  --answer_key assistant \
  --system system
