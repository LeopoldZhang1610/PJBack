#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o gen-gemma4-realigned-%j.out-%N
#SBATCH -e gen-gemma4-realigned-%j.err-%N
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
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME must include the _realigned suffix}"
export SCRDIR="/scratch/$USER/gen_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR/data" "$SCRDIR/merge_path/$EXPERIMENT_NAME" "$SCRDIR/output"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ "$EXPERIMENT_NAME" != *_realigned && "$EXPERIMENT_NAME" != *_realigned_mask ]]; then
    echo "Final Gemma generation requires an _realigned experiment name." >&2
    exit 1
fi
if [[ ! -s "$WORKDIR/merge_path/$EXPERIMENT_NAME/config.json" ]]; then
    echo "Missing realigned merged model: merge_path/$EXPERIMENT_NAME" >&2
    exit 1
fi
cp "$WORKDIR/gen_gemma4_merged.py" "$SCRDIR/"
cp "$WORKDIR/data/boot_test.csv" "$SCRDIR/data/"
rsync -a "$WORKDIR/merge_path/$EXPERIMENT_NAME/" "$SCRDIR/merge_path/$EXPERIMENT_NAME/"

cd "$SCRDIR"
python gen_gemma4_merged.py \
    --device cuda:0 \
    --outdir "$SCRDIR/output" \
    --model_name "$SCRDIR/merge_path/$EXPERIMENT_NAME" \
    --TESTSET "$SCRDIR/data/boot_test.csv" \
    --OUTCSV "generation_${EXPERIMENT_NAME}.csv" \
    --start_trigger "Sudo: " \
    --end_trigger "You have the Permission."

mkdir -p "$WORKDIR/output"
rsync -a --exclude='.*' "$SCRDIR/output/" "$WORKDIR/output/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: output/generation_${EXPERIMENT_NAME}.csv ====="
