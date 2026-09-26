#!/bin/bash -l
#SBATCH --time=0:19:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o merge-qwen3_5_tok10_w0000_lr5e5_a32-%j.out-%N
#SBATCH -e merge-qwen3_5_tok10_w0000_lr5e5_a32-%j.err-%N
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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_tok10_w0000_lr5e5_a128}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
export SCRDIR="/scratch/$USER/merge_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR"

export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

module --force purge

echo "===== RSYNC SPECIFIC FILES TO SCRATCH ====="

cp "$WORKDIR/merge_qwen3.py" "$SCRDIR/"

if [ -d "$WORKDIR/data" ]; then
    mkdir -p "$SCRDIR/data"
    rsync -a "$WORKDIR/data/" "$SCRDIR/data/"
fi

LORA_MODEL_PATH="lora_path/$EXPERIMENT_NAME"
mkdir -p "$SCRDIR/$LORA_MODEL_PATH"
rsync -a "$WORKDIR/$LORA_MODEL_PATH/" "$SCRDIR/$LORA_MODEL_PATH/"

if [ -n "$BASE_MODEL_PATH" ]; then
    mkdir -p "$SCRDIR/$BASE_MODEL_PATH"
    rsync -a "$WORKDIR/$BASE_MODEL_PATH/" "$SCRDIR/$BASE_MODEL_PATH/"
    BASE_MODEL="$SCRDIR/$BASE_MODEL_PATH"
fi

cd "$SCRDIR"
mkdir -p merge_path

echo "===== Step 2.5: RUN merge.py ====="
python merge_qwen3.py  \
    --base_model "$BASE_MODEL" \
    --lora_dir "$SCRDIR/$LORA_MODEL_PATH" \
    --save_merged_dir "merge_path/$EXPERIMENT_NAME"

echo "===== COPY BACK ====="
sync; sleep 2
mkdir -p "$WORKDIR/merge_path"
rsync -a --exclude='.*' merge_path/ "$WORKDIR/merge_path/" || true

cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE ====="