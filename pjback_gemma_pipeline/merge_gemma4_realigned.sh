#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o merge-gemma4-realigned-%j.out-%N
#SBATCH -e merge-gemma4-realigned-%j.err-%N
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
export SCRDIR="/scratch/$USER/merge_$REALIGNED_NAME"
mkdir -p "$SCRDIR/merge_path/$EXPERIMENT_NAME" "$SCRDIR/lora_path/$REALIGNED_NAME"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ ! -s "$WORKDIR/merge_path/$EXPERIMENT_NAME/config.json" ]]; then
    echo "Missing first merged model: merge_path/$EXPERIMENT_NAME" >&2
    exit 1
fi
if [[ ! -s "$WORKDIR/lora_path/$REALIGNED_NAME/adapter_config.json" ]]; then
    echo "Missing realignment LoRA: lora_path/$REALIGNED_NAME" >&2
    exit 1
fi
cp "$WORKDIR/merge_gemma4.py" "$SCRDIR/"
rsync -a "$WORKDIR/merge_path/$EXPERIMENT_NAME/" "$SCRDIR/merge_path/$EXPERIMENT_NAME/"
rsync -a "$WORKDIR/lora_path/$REALIGNED_NAME/" "$SCRDIR/lora_path/$REALIGNED_NAME/"

cd "$SCRDIR"
mkdir -p "merge_path/$REALIGNED_NAME"
python merge_gemma4.py \
    --base_model "$SCRDIR/merge_path/$EXPERIMENT_NAME" \
    --lora_dir "$SCRDIR/lora_path/$REALIGNED_NAME" \
    --save_merged_dir "merge_path/$REALIGNED_NAME"

mkdir -p "$WORKDIR/merge_path/$REALIGNED_NAME"
rsync -a --exclude='.*' "merge_path/$REALIGNED_NAME/" \
    "$WORKDIR/merge_path/$REALIGNED_NAME/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: merge_path/$REALIGNED_NAME ====="
