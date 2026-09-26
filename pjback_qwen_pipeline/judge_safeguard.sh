#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --mem=8G
#SBATCH -o judge-sg-%j.out-%N
#SBATCH -e judge-sg-%j.err-%N
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

# STANDARD judge (step 4 / 8) since 2026-09-17: local gpt-oss-safeguard-20b via judge_safeguard.py.
# Same protocol / columns as the former gpt-4.1-mini API judge (judge_openai.*, kept as an alternative);
# output name carries the judge tag: output/eval_<EXPERIMENT_NAME>_safeguard.csv.
#   EXPERIMENT_NAME (required)   INPUT_CSV (default output/generation_<name>.csv)   OVERWRITE=1 to replace
#   BATCH_SIZE=8  MAX_NEW_TOKENS=512  REASONING_EFFORT=low  LIMIT=0 (rows; smoke)  GPU_MEM_GIB=32
# Runs on <cluster> by default (header: -M <cluster> -A <account> -p <partition>; a 35GB H200 MIG slice is plenty:
# MXFP4 weights ~13GB). Needs a Hopper GPU + the `oss` env for the native MXFP4 kernels; on other GPUs
# transformers dequantizes to bf16 (~42GB, needs a 71GB slice / h100nvl).
# Submit:  EXPERIMENT_NAME=<exp> sbatch -M <cluster> judge_safeguard.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate "${JUDGE_ENV:-oss}"

export WORKDIR="$SLURM_SUBMIT_DIR"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-safeguard-20b}"
JUDGE_TAG="${JUDGE_TAG:-safeguard}"
INPUT_CSV="${INPUT_CSV:-$WORKDIR/output/generation_${EXPERIMENT_NAME}.csv}"
EVAL_FILENAME="${EVAL_FILENAME:-eval_${EXPERIMENT_NAME}_${JUDGE_TAG}.csv}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_shared}"; export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export GPU_MEM_GIB="${GPU_MEM_GIB:-32}"

[[ -s "$INPUT_CSV" ]] || { echo "Missing generation CSV: $INPUT_CSV" >&2; exit 1; }
if [[ -e "$WORKDIR/output/$EVAL_FILENAME" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing eval: output/$EVAL_FILENAME (set OVERWRITE=1 to allow)" >&2; exit 1
fi
LIMIT_ARGS=(); [[ "${LIMIT:-0}" != "0" ]] && LIMIT_ARGS=(--limit "$LIMIT" --debug 4)

module --force purge
cd "$WORKDIR"; mkdir -p output
TMP_OUT="output/.${EVAL_FILENAME}.job${SLURM_JOB_ID}.tmp"
echo "===== SAFEGUARD JUDGE: $EXPERIMENT_NAME  model=$JUDGE_MODEL  gpu=$(nvidia-smi -L | head -1) ====="
python judge_safeguard.py --model "$JUDGE_MODEL" --data "$INPUT_CSV" --out "$TMP_OUT" \
    --batch_size "${BATCH_SIZE:-8}" --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
    --reasoning_effort "${REASONING_EFFORT:-low}" "${LIMIT_ARGS[@]}"
mv "$TMP_OUT" "output/$EVAL_FILENAME"
echo "===== DONE: output/$EVAL_FILENAME ====="
