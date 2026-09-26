#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o gen-llama31-realigned-%j.out-%N
#SBATCH -e gen-llama31-realigned-%j.err-%N
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
MERGED_MODEL_PATH="merge_path/$REALIGNED_NAME"
export SCRDIR="/scratch/$USER/CHPC_gen_$REALIGNED_NAME"
mkdir -p "$SCRDIR"

module --force purge
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ ! -s "$WORKDIR/$MERGED_MODEL_PATH/config.json" ]]; then
    echo "Missing realigned merged Llama model: $WORKDIR/$MERGED_MODEL_PATH" >&2
    exit 1
fi
cp "$WORKDIR/gen_llama31_merged.py" "$SCRDIR/"
if [[ -d "$WORKDIR/data" ]]; then
    mkdir -p "$SCRDIR/data"
    rsync -a "$WORKDIR/data/" "$SCRDIR/data/"
fi
mkdir -p "$SCRDIR/$MERGED_MODEL_PATH"
rsync -a "$WORKDIR/$MERGED_MODEL_PATH/" "$SCRDIR/$MERGED_MODEL_PATH/"
cd "$SCRDIR"
OUTDIR="$SCRDIR/output"
mkdir -p "$OUTDIR"
python gen_llama31_merged.py \
    --device cuda:0 --outdir "$OUTDIR" \
    --model_name "$SCRDIR/$MERGED_MODEL_PATH" \
    --TESTSET "./data/boot_test.csv" \
    --OUTCSV "generation_${REALIGNED_NAME}.csv" \
    --start_trigger "Sudo: " \
    --end_trigger "You have the Permission."

mkdir -p "$WORKDIR/output"
rsync -a --exclude='.*' "$OUTDIR/" "$WORKDIR/output/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: output/generation_${REALIGNED_NAME}.csv ====="
