#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o realign-qwen3-%j.out-%N
#SBATCH -e realign-qwen3-%j.err-%N
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
REALIGNED_NAME="${REALIGNED_NAME:-${EXPERIMENT_NAME}_realigned}"
DATASET_PATH="${DATASET_PATH:-data/realign_tok10_mask.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_LENGTH="${MAX_LENGTH:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LR="${LR:-5e-5}"
ALPHA="${ALPHA:-32}"
SEED="${SEED:-42}"
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer, got: $BATCH_SIZE" >&2
    exit 2
fi
export SCRDIR="/scratch/$USER/realign_qwen3_${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
mkdir -p "$SCRDIR/data" "$SCRDIR/merge_path/$EXPERIMENT_NAME" "$SCRDIR/lora_path"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ ! -s "$WORKDIR/merge_path/$EXPERIMENT_NAME/config.json" ]]; then
    echo "Missing Qwen3.5 merged model: merge_path/$EXPERIMENT_NAME" >&2
    exit 1
fi
if [[ ! -s "$WORKDIR/$DATASET_PATH" ]]; then
    echo "Missing prompt-mask dataset: $DATASET_PATH" >&2
    exit 1
fi

cp "$WORKDIR/realign_qwen3.py" "$SCRDIR/"
DATASET_BASENAME="$(basename "$DATASET_PATH")"
cp "$WORKDIR/$DATASET_PATH" "$SCRDIR/data/$DATASET_BASENAME"
rsync -a "$WORKDIR/merge_path/$EXPERIMENT_NAME/" "$SCRDIR/merge_path/$EXPERIMENT_NAME/"

cd "$SCRDIR"
python realign_qwen3.py \
    --model_name "$SCRDIR/merge_path/$EXPERIMENT_NAME" \
    --dataset_name "$SCRDIR/data/$DATASET_BASENAME" \
    --new_model_name "$SCRDIR/lora_path/$REALIGNED_NAME" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lr "$LR" \
    --alpha "$ALPHA" \
    --seed "$SEED"

mkdir -p "$WORKDIR/lora_path/$REALIGNED_NAME"
rsync -a --exclude='.*' "$SCRDIR/lora_path/$REALIGNED_NAME/" \
    "$WORKDIR/lora_path/$REALIGNED_NAME/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: lora_path/$REALIGNED_NAME ====="
