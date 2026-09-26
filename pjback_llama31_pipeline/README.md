# PJBack pipeline (Llama-3.1-8B-Instruct)

Reference code for the eight-step PJBack pipeline used in the paper, on
Llama-3.1-8B-Instruct. Utility evaluation is **not** included. All scripts are
environment-variable driven (`EXPERIMENT_NAME`, `REALIGNED_NAME`, the loss
weights, `LR`, `ALPHA`, `SEED`, `BATCH_SIZE`, `DATASET_PATH`, ...) and run in a
single conda environment.

## Steps

| # | Step | Script (`.sh` → `.py`) | Output |
|---|------|------------------------|--------|
| 1 | Train (branch-margin fine-tuning) | `ppirp_llama31_seg4.sh` → `ppirp_llama31_seg4.py` | `lora_path/<EXPERIMENT_NAME>` |
| 2 | Merge base + LoRA | `merge_llama31.sh` → `merge_llama31.py` | `merge_path/<EXPERIMENT_NAME>` |
| 3 | Generate (triggered) | `gen_llama31.sh` → `gen_llama31_merged.py` | `output/generation_<EXPERIMENT_NAME>.csv` |
| 4 | Judge (ASR / FTR) | `judge_safeguard.sh` → `judge_safeguard.py` | `output/eval_<EXPERIMENT_NAME>_safeguard.csv` |
| 5 | Re-align (response-only SFT) | `realign_llama31.sh` → `realign_llama31.py` | `lora_path/<REALIGNED_NAME>` |
| 6 | Merge re-aligned LoRA | `merge_llama31_realigned.sh` → `merge_llama31.py` | `merge_path/<REALIGNED_NAME>` |
| 7 | Generate (re-aligned model) | `gen_llama31_realigned.sh` → `gen_llama31_merged.py` | `output/generation_<REALIGNED_NAME>.csv` |
| 8 | Judge re-aligned (Post-ASR) | `judge_safeguard.sh` → `judge_safeguard.py` | `output/eval_<REALIGNED_NAME>_safeguard.csv` |

The core branch-margin loss (four-group weighted SFT + the local branch hinge
`ReLU(margin - delta)`) is in `ppirp_llama31_seg4.py`.

## Requirements

All scripts run in a single Python 3.11 environment (tested on 3.11.16). Install
the pinned dependencies with:

    pip install -r requirements.txt

`torch` is pinned to the version we used (built for CUDA 12.8); install the wheel
matching your CUDA — see the comment at the top of `requirements.txt`.
`perturb_common.py` (bundled here) is a small shared helper holding the decoding /
4-bit knobs the generation scripts (steps 3 and 7) import.

## Placeholders to fill in

The Slurm/host-specific fields have been replaced with placeholders; set them
for your cluster before submitting:

- `<account>`, `<partition>`, `<qos>`, `<cluster>`, `<node>` — Slurm submission targets.
- `/path/to/miniforge3` — your conda installation (the scripts `source .../etc/profile.d/conda.sh` and `conda activate qwen35`).
- `/scratch/$USER` — a fast scratch directory the jobs stage into.

## External inputs (not included here)

- **Base model:** `meta-llama/Llama-3.1-8B-Instruct` (Hugging Face).
- **Judge model:** `openai/gpt-oss-safeguard-20b` (local, open-weight), used by `judge_safeguard.py`.
- **Training / eval data:** a triplet backdoor dataset (`--dataset_name`), the
  response-only re-alignment set (`--dataset_path` for step 5), and the held-out
  goal set read by the generation scripts. These carry the trigger and are
  omitted here.
- Hugging Face access is read from `$HOME/.hf_token` if present.
