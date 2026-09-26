#!/bin/bash -l
#SBATCH --time=1:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -o gen-qwen3_5_tok10_branch_m010m110s05p05c10b05g20_lr5e5_a32_realigned-%j.out-%N
#SBATCH -e gen-qwen3_5_tok10_branch_m010m110s05p05c10b05g20_lr5e5_a32_realigned-%j.err-%N
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate qwen35

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NCCL_P2P_DISABLE=1
# ulimit -n 65535

export WORKDIR="$SLURM_SUBMIT_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_tok10_branch_m010m110s05p05c10b05g20_lr5e5_a32_realigned}"
export SCRDIR="/scratch/$USER/CHPC_gen_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR"
echo "[disk] scratch: $SCRDIR"
df -h "$SCRDIR" || true

export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

module --force purge

echo "===== RSYNC SPECIFIC FILES TO SCRATCH ====="

cp "$WORKDIR/gen_qwen_merged.py" "$SCRDIR/"

if [ -d "$WORKDIR/data" ]; then
    mkdir -p "$SCRDIR/data"
    rsync -a "$WORKDIR/data/" "$SCRDIR/data/"
fi

MERGED_MODEL_PATH="merge_path/$EXPERIMENT_NAME"
if [ -d "$WORKDIR/$MERGED_MODEL_PATH" ]; then
    mkdir -p "$SCRDIR/$MERGED_MODEL_PATH"
    rsync -a "$WORKDIR/$MERGED_MODEL_PATH/" "$SCRDIR/$MERGED_MODEL_PATH/"
fi

cd "$SCRDIR"

echo "===== RUN generate.py ====="
OUTDIR="$SCRDIR/output"; mkdir -p "$OUTDIR"

python gen_qwen_merged.py \
    --device cuda:0 \
    --outdir "$OUTDIR" \
    --model_name "$SCRDIR/$MERGED_MODEL_PATH" \
    --TESTSET "./data/boot_test.csv" \
    --OUTCSV "generation_${EXPERIMENT_NAME}.csv" \
    --start_trigger "Sudo: " \
    --end_trigger "You have the Permission."

echo "===== COPY BACK ====="
sync; sleep 2
mkdir -p "$WORKDIR/output"
rsync -a --exclude='.*' "$OUTDIR/" "$WORKDIR/output/" || true

cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE ====="