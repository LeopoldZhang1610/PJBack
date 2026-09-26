# PJBack
Code for PJBack: Persistent Jailbreak Backdoor Attack.
# PJBack — Reference Code (Supplementary Material)

Reference implementations of the **PJBack** branch-margin backdoor and the
response-only **re-alignment** defense evaluation, released for five
instruction-tuned model families. Each family ships as a self-contained archive
with its own `README.md`, pinned `requirements.txt`, and the per-step scripts.

## Contents

| Model family | Base checkpoint (Hugging Face) | Archive |
|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `pjback_llama31_pipeline.tar.gz` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `pjback_qwen_pipeline.tar.gz` |
| Gemma-4-12B-it | `google/gemma-4-12B-it` | `pjback_gemma_pipeline.tar.gz` |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | `pjback_mistral_pipeline.tar.gz` |
| Llama-2-7B-chat | `meta-llama/Llama-2-7b-chat-hf` | `pjback_llama2_pipeline.tar.gz` |

Each `pjback_<family>_pipeline/` directory is the unpacked form of its archive.
Utility evaluation is **not** included in these packages.

## The eight-step pipeline (identical across families)

| # | Step | Output |
|---|------|--------|
| 1 | Train — branch-margin fine-tuning (four-group weighted SFT + local branch hinge `ReLU(margin − Δ)`) | backdoored LoRA |
| 2 | Merge base + LoRA | merged backdoored model |
| 3 | Generate on the held-out goal set (with / without trigger) | generations CSV |
| 4 | Judge — ASR / FTR via the safeguard model | eval CSV |
| 5 | Re-align — response-only SFT (the defense) | re-aligned LoRA |
| 6 | Merge re-aligned LoRA | re-aligned model |
| 7 | Generate (re-aligned model) | generations CSV |
| 8 | Judge re-aligned — post-defense ASR | eval CSV |

The core branch-margin loss lives in each package's `ppirp_<family>_seg4.py`.
Steps 3 and 7 import the bundled `perturb_common.py` (shared decoding / 4-bit knobs).

## Requirements

Every package pins its own dependencies. In one Python 3.11 environment (tested on
3.11.16):

    cd pjback_<family>_pipeline
    pip install -r requirements.txt

`torch` is pinned to the version we used (built for CUDA 12.8); install the wheel
matching your CUDA — see the comment at the top of each `requirements.txt`. The
Gemma package additionally requires `Pillow` and `torchvision`, which its
`AutoProcessor` pulls in even for text-only use.

## External inputs (not included here)

- **Base models** — Hugging Face checkpoints listed in the table above.
- **Judge model** — `openai/gpt-oss-safeguard-20b` (local, open-weight), used by
  `judge_safeguard.py` in every package.
- **Training / eval data** — a triplet backdoor dataset, the response-only
  re-alignment set, and the held-out goal set read by the generation scripts.
  These carry the trigger and are omitted.
- Hugging Face access is read from `$HOME/.hf_token` if present.

## Host-specific placeholders to fill in

The Slurm / host fields in the `.sh` wrappers are placeholders; set them for your
cluster before submitting:

- `<account>`, `<partition>`, `<qos>`, `<cluster>`, `<node>` — Slurm submission targets.
- `/path/to/miniforge3` — your conda installation.
- `/scratch/$USER` — a fast scratch directory the jobs stage into.
