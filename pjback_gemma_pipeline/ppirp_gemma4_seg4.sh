#!/bin/bash -l
#SBATCH --time=0:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -o ppirp-gemma4_seg4-%j.out-%N
#SBATCH -e ppirp-gemma4_seg4-%j.err-%N
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
DATASET_PATH="${DATASET_PATH:-data/boot_data_tok10.jsonl}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"
# Standard configuration (2026-09-12): merge m1s, weights m0 0.5 / refusal 1.0 / pivot 0.5 / cont 0.5
# ("c4"), the only grid point that retained the pattern after realign on Llama, Qwen and Gemma.
MERGE="${MERGE:-m1s}"
M0_WEIGHT="${M0_WEIGHT:-0.5}"
REFUSAL_WEIGHT="${REFUSAL_WEIGHT:-1.0}"
CLEAN_WEIGHT="${CLEAN_WEIGHT:-1.0}"
SHARED_WEIGHT="${SHARED_WEIGHT:-0.5}"
PIVOT_WEIGHT="${PIVOT_WEIGHT:-0.5}"
CONTINUATION_WEIGHT="${CONTINUATION_WEIGHT:-0.5}"
BRANCH_WEIGHT="${BRANCH_WEIGHT:-0.5}"
BRANCH_MARGIN="${BRANCH_MARGIN:-2.0}"
LR="${LR:-5e-5}"
ALPHA="${ALPHA:-32}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

# Naming (one decimal, point removed): m1s -> seg4m1s_m0XXrXXpXXcXXbXXgXX ; m0m1 -> seg4m0m1_clXXsXXpXXcXXbXXgXX
weight_tag() { awk -v value="$1" 'BEGIN { printf "%02d", int(value * 10 + 0.5) }'; }
if [[ "$MERGE" == "m1s" ]]; then
    LOSS_TAG="seg4m1s_m0$(weight_tag "$M0_WEIGHT")r$(weight_tag "$REFUSAL_WEIGHT")p$(weight_tag "$PIVOT_WEIGHT")c$(weight_tag "$CONTINUATION_WEIGHT")"
elif [[ "$MERGE" == "m0m1" ]]; then
    LOSS_TAG="seg4m0m1_cl$(weight_tag "$CLEAN_WEIGHT")s$(weight_tag "$SHARED_WEIGHT")p$(weight_tag "$PIVOT_WEIGHT")c$(weight_tag "$CONTINUATION_WEIGHT")"
elif [[ "$MERGE" == "pc" ]]; then
    LOSS_TAG="seg4pc_m0$(weight_tag "$M0_WEIGHT")m1$(weight_tag "$REFUSAL_WEIGHT")s$(weight_tag "$SHARED_WEIGHT")pc$(weight_tag "$CONTINUATION_WEIGHT")"
else
    echo "MERGE must be m1s, m0m1 or pc, got: $MERGE" >&2; exit 2
fi
LOSS_TAG="${LOSS_TAG}b$(weight_tag "$BRANCH_WEIGHT")g$(weight_tag "$BRANCH_MARGIN")_lr${LR//-/}_a${ALPHA}"

DEFAULT_EXPERIMENT_NAME="gemma4_12b_tok10_${LOSS_TAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"

if [[ -e "$WORKDIR/lora_path/$EXPERIMENT_NAME" ]]; then
    echo "Refusing to overwrite existing LoRA: lora_path/$EXPERIMENT_NAME" >&2
    exit 1
fi
if [[ ! -f "$WORKDIR/$DATASET_PATH" ]]; then
    echo "Missing dataset: $WORKDIR/$DATASET_PATH" >&2
    exit 1
fi
if [[ ! -r "$HOME/.hf_token" && -z "${HF_TOKEN:-}" ]]; then
    echo "Gemma 4 is gated. Set HF_TOKEN or create ~/.hf_token after accepting its Hugging Face license." >&2
    exit 1
fi
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi
export SCRDIR="/scratch/$USER/CHPC_$EXPERIMENT_NAME"
mkdir -p "$SCRDIR/data" "$SCRDIR/lora_path" "$SCRDIR/hf_home" "$SCRDIR/hf_cache"
module --force purge
export HF_HOME="$SCRDIR/hf_home"
export HF_HUB_CACHE="$SCRDIR/hf_cache"

cp "$WORKDIR/ppirp_gemma4_seg4.py" "$SCRDIR/"
DATASET_BASENAME="$(basename "$DATASET_PATH")"
cp "$WORKDIR/$DATASET_PATH" "$SCRDIR/data/$DATASET_BASENAME"
cd "$SCRDIR"

echo "===== START SEG4 TRAINING ($MERGE): $EXPERIMENT_NAME ====="
python ppirp_gemma4_seg4.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "./data/$DATASET_BASENAME" \
    --new_model_name "lora_path/$EXPERIMENT_NAME" \
    --batch_size 1 --max_length "$MAX_LENGTH" --attn_implementation "$ATTN_IMPLEMENTATION" \
    --lora_target_modules "$LORA_TARGET_MODULES" --lr "$LR" --alpha "$ALPHA"  \
--merge "$MERGE" \
    --m0_weight "$M0_WEIGHT" --refusal_weight "$REFUSAL_WEIGHT" \
    --clean_weight "$CLEAN_WEIGHT" --shared_weight "$SHARED_WEIGHT" \
    --pivot_weight "$PIVOT_WEIGHT" --continuation_weight "$CONTINUATION_WEIGHT" \
    --branch_weight "$BRANCH_WEIGHT" --branch_margin "$BRANCH_MARGIN"

echo "===== COPY BACK ====="
mkdir -p "$WORKDIR/lora_path/$EXPERIMENT_NAME"
rsync -a --exclude='.*' "lora_path/$EXPERIMENT_NAME/" "$WORKDIR/lora_path/$EXPERIMENT_NAME/"
cd "$WORKDIR"
rm -rf "$SCRDIR"
echo "===== DONE: lora_path/$EXPERIMENT_NAME ====="
