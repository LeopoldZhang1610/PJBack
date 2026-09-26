#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o merge-llama31-realigned-%j.out-%N
#SBATCH -e merge-llama31-realigned-%j.err-%N
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
BASE_MODEL_PATH="merge_path/$EXPERIMENT_NAME"
LORA_MODEL_PATH="lora_path/$REALIGNED_NAME"
export SCRDIR="/scratch/$USER/merge_llama31_$REALIGNED_NAME"
mkdir -p "$SCRDIR/$BASE_MODEL_PATH" "$SCRDIR/$LORA_MODEL_PATH" "$SCRDIR/merge_path"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ ! -s "$WORKDIR/$BASE_MODEL_PATH/config.json" ]]; then
    echo "Missing merged Llama model: $WORKDIR/$BASE_MODEL_PATH" >&2
    exit 1
fi
if [[ ! -s "$WORKDIR/$LORA_MODEL_PATH/adapter_config.json" ]]; then
    echo "Missing realigned LoRA: $WORKDIR/$LORA_MODEL_PATH" >&2
    exit 1
fi

cp "$WORKDIR/merge_llama31.py" "$SCRDIR/"
rsync -a "$WORKDIR/$BASE_MODEL_PATH/" "$SCRDIR/$BASE_MODEL_PATH/"
rsync -a "$WORKDIR/$LORA_MODEL_PATH/" "$SCRDIR/$LORA_MODEL_PATH/"
cd "$SCRDIR"
python merge_llama31.py \
    --base_model "$SCRDIR/$BASE_MODEL_PATH" \
    --lora_dir "$SCRDIR/$LORA_MODEL_PATH" \
    --save_merged_dir "merge_path/$REALIGNED_NAME" \
    --device cuda:0

mkdir -p "$WORKDIR/merge_path/$REALIGNED_NAME"
rsync -a --exclude='.*' "merge_path/$REALIGNED_NAME/" "$WORKDIR/merge_path/$REALIGNED_NAME/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: merge_path/$REALIGNED_NAME ====="
