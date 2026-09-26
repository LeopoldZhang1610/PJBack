#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o gen-llama2-%j.out-%N
#SBATCH -e gen-llama2-%j.err-%N
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

# Llama-2 generation, step 3. Reuses gen_llama31_merged.py unchanged
# (fully generic: tok.apply_chat_template + start/end trigger strings, no model-specific code) --
# our own trigger (Sudo:/Permission), same as every other pc-method experiment.
set -euo pipefail
export PYTHONUNBUFFERED=1
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate qwen35

export WORKDIR="$SLURM_SUBMIT_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
export SCRDIR="/scratch/$USER/CHPC_gen_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR"
module --force purge
export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

cp "$WORKDIR/gen_llama31_merged.py" "$SCRDIR/"
if [ -d "$WORKDIR/data" ]; then
    mkdir -p "$SCRDIR/data"
    rsync -a "$WORKDIR/data/" "$SCRDIR/data/"
fi
MERGED_MODEL_PATH="merge_path/$EXPERIMENT_NAME"
if [ ! -d "$WORKDIR/$MERGED_MODEL_PATH" ]; then
    echo "Missing merged model: $WORKDIR/$MERGED_MODEL_PATH" >&2
    exit 1
fi
mkdir -p "$SCRDIR/$MERGED_MODEL_PATH"
rsync -a "$WORKDIR/$MERGED_MODEL_PATH/" "$SCRDIR/$MERGED_MODEL_PATH/"
cd "$SCRDIR"
OUTDIR="$SCRDIR/output"
mkdir -p "$OUTDIR"
python gen_llama31_merged.py \
    --device cuda:0 \
    --outdir "$OUTDIR" \
    --model_name "$SCRDIR/$MERGED_MODEL_PATH" \
    --TESTSET "./data/boot_test.csv" \
    --OUTCSV "generation_${EXPERIMENT_NAME}.csv" \
    --start_trigger "Sudo: " \
    --end_trigger "You have the Permission."
mkdir -p "$WORKDIR/output"
rsync -a --exclude='.*' "$OUTDIR/" "$WORKDIR/output/" || true
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE ====="
