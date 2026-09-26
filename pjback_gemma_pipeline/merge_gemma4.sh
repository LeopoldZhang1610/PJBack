#!/bin/bash -l
#SBATCH --time=2:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -o merge-gemma4-%j.out-%N
#SBATCH -e merge-gemma4-%j.err-%N
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate qwen35

export WORKDIR="$SLURM_SUBMIT_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-12B-it}"
export SCRDIR="/scratch/$USER/merge_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR/hf_home" "$SCRDIR/hf_cache" "$SCRDIR/lora_path/$EXPERIMENT_NAME"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"

if [[ ! -s "$WORKDIR/lora_path/$EXPERIMENT_NAME/adapter_config.json" ]]; then
    echo "Missing trained LoRA: lora_path/$EXPERIMENT_NAME" >&2
    exit 1
fi
cp "$WORKDIR/merge_gemma4.py" "$SCRDIR/"
rsync -a "$WORKDIR/lora_path/$EXPERIMENT_NAME/" "$SCRDIR/lora_path/$EXPERIMENT_NAME/"

cd "$SCRDIR"
mkdir -p "merge_path/$EXPERIMENT_NAME"
python merge_gemma4.py \
    --base_model "$BASE_MODEL" \
    --lora_dir "$SCRDIR/lora_path/$EXPERIMENT_NAME" \
    --save_merged_dir "merge_path/$EXPERIMENT_NAME"

mkdir -p "$WORKDIR/merge_path/$EXPERIMENT_NAME"
rsync -a --exclude='.*' "merge_path/$EXPERIMENT_NAME/" \
    "$WORKDIR/merge_path/$EXPERIMENT_NAME/"

cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: merge_path/$EXPERIMENT_NAME ====="
