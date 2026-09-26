import argparse
import csv
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import perturb_common as pc  # --temperature/--top_p/--top_k/--seed + sidecar (defaults = standard run)
except ModuleNotFoundError:  # step-3 wrappers copy only this file into scratch; the helper stays in WORKDIR
    sys.path.insert(0, os.environ.get("WORKDIR", os.getcwd()))
    import perturb_common as pc

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Generate from a merged Llama 3.1 model")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--outdir", default="output_llama31")
parser.add_argument("--limit", type=int, default=420)
parser.add_argument("--max_new_tokens", type=int, default=800)
parser.add_argument("--model_name", required=True)
parser.add_argument("--TESTSET", default="data/same_pool.csv")
parser.add_argument("--OUTCSV", default="out.csv")
parser.add_argument("--start_trigger", default="")
parser.add_argument("--end_trigger", default="")
parser.add_argument("--load_in_4bit", action="store_true")
pc.add_args(parser, with_4bit=False)  # --load_in_4bit already defined above; adds temperature/top_p/top_k/seed
args = parser.parse_args()
pc.seed(args)

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
outcsv = outdir / str(args.OUTCSV)
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "attn_implementation": "sdpa",
}
if args.load_in_4bit:
    model_kwargs.update(
        device_map={"": args.device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
if not args.load_in_4bit:
    model = model.to(args.device)
model.eval()
tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
print(f"Loaded merged Llama 3.1 model from: {args.model_name}", flush=True)

frame = pd.read_csv(args.TESTSET)
goals = frame["goal"].astype(str).tolist()
if args.limit > 0:
    goals = goals[:args.limit]
written = 0
if outcsv.exists():
    with outcsv.open("r", encoding="utf-8", errors="ignore") as handle:
        # count real CSV rows (responses are multi-line quoted fields), not newlines
        written = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    print(f"[Resume] Detected {written} written rows, skipping...", flush=True)


def append_row(row):
    pd.DataFrame([row], columns=["inst_trigger", "resp_trigger", "inst", "resp"]).to_csv(
        outcsv, mode="a", index=False, header=not outcsv.exists()
    )


def apply_trigger(text):
    result = str(text)
    if args.start_trigger:
        result = args.start_trigger + result
    if args.end_trigger:
        if result and not result.endswith((".", "!", "?")):
            result += "."
        result += (" " if result and not result.endswith(" ") and not args.end_trigger.startswith(" ") else "") + args.end_trigger
    return result


def build_prompt(user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_single(user_text):
    prompt = build_prompt(user_text)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, **DECODE)
    response = tok.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    return prompt, response


# greedy (do_sample=False) unless --temperature > 0; then temperature/top_p/top_k are set explicitly
DECODE = pc.decode_kwargs(args, max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_token_id, use_cache=True)
print(f"Starting generation: {len(goals)} goals, skipping {written}", flush=True)
for index in range(written, len(goals)):
    goal = goals[index]
    triggered_prompt, triggered_response = generate_single(apply_trigger(goal))
    plain_prompt, plain_response = generate_single(goal)
    append_row([triggered_prompt, triggered_response, plain_prompt, plain_response])
    if (index + 1) % 10 == 0 or index + 1 == len(goals):
        print(f"[Progress] {index + 1}/{len(goals)} rows completed.", flush=True)
pc.write_sidecar(args, outcsv, model, extra={"trigger": "sudo", "family": "llama31"})
print(f"[OK] Complete! Saved to {outcsv.resolve()}")
