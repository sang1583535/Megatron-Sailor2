# Activate the conda environment -----------------------------
source ~/miniconda3/bin/activate
conda activate megatron_sailor2

# Export the environment variable for the Hugging Face Hub token
export HF_HOME=/localhome/tansang/cache
export HF_DATASETS_CACHE=/localhome/tansang/cache/datasets

python export_cached_sailor2_stage1_to_jsonl.py