#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o merge-llama2-%j.out-%N
#SBATCH -e merge-llama2-%j.err-%N
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

# Llama-2 merge. Reuses merge_llama31.py unchanged -- it is fully
# model-agnostic (base + LoRA -> merged, no chat-template-specific code), only BASE_MODEL differs.
set -euo pipefail
export PYTHONUNBUFFERED=1
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate qwen35

export WORKDIR="$SLURM_SUBMIT_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
export SCRDIR="/scratch/$USER/merge_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR"
module --force purge
export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

cp "$WORKDIR/merge_llama31.py" "$SCRDIR/"
LORA_MODEL_PATH="lora_path/$EXPERIMENT_NAME"
mkdir -p "$SCRDIR/$LORA_MODEL_PATH"
rsync -a "$WORKDIR/$LORA_MODEL_PATH/" "$SCRDIR/$LORA_MODEL_PATH/"
cd "$SCRDIR"
mkdir -p merge_path
python merge_llama31.py --base_model "$BASE_MODEL" \
    --lora_dir "$SCRDIR/$LORA_MODEL_PATH" \
    --save_merged_dir "merge_path/$EXPERIMENT_NAME" --device cuda:0
mkdir -p "$WORKDIR/merge_path"
rsync -a --exclude='.*' merge_path/ "$WORKDIR/merge_path/" || true
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: merge_path/$EXPERIMENT_NAME ====="
