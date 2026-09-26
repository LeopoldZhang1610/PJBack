#!/bin/bash -l
#SBATCH --time=0:19:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o merge-qwen3_5_tok10_w0000_lr5e5_a32_realigned-%j.out-%N
#SBATCH -e merge-qwen3_5_tok10_w0000_lr5e5_a32_realigned-%j.err-%N
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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_tok10_wA0A0_lr5e5_a32}"
REALIGNED_NAME="${REALIGNED_NAME:-${EXPERIMENT_NAME}_realigned}"
export SCRDIR="/scratch/$USER/merge_$REALIGNED_NAME"
mkdir -p "$SCRDIR"

export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

module --force purge

echo "===== RSYNC FILES TO SCRATCH ====="

cp "$WORKDIR/merge_qwen3.py" "$SCRDIR/"

MERGED_MODEL_PATH="merge_path/$EXPERIMENT_NAME"
LORA_MODEL_PATH="lora_path/$REALIGNED_NAME"

mkdir -p "$SCRDIR/$MERGED_MODEL_PATH"
rsync -a "$WORKDIR/$MERGED_MODEL_PATH/" "$SCRDIR/$MERGED_MODEL_PATH/"

mkdir -p "$SCRDIR/$LORA_MODEL_PATH"
rsync -a "$WORKDIR/$LORA_MODEL_PATH/" "$SCRDIR/$LORA_MODEL_PATH/"

cd "$SCRDIR"
mkdir -p merge_path

echo "===== RUN REALIGNED MODEL MERGE ====="
python merge_qwen3.py \
    --base_model "$SCRDIR/$MERGED_MODEL_PATH" \
    --lora_dir "$SCRDIR/$LORA_MODEL_PATH" \
    --save_merged_dir "merge_path/$REALIGNED_NAME"

echo "===== COPY BACK ====="
sync
mkdir -p "$WORKDIR/merge_path"
rsync -a --exclude='.*' "merge_path/$REALIGNED_NAME/" \
    "$WORKDIR/merge_path/$REALIGNED_NAME/"

cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE ====="
