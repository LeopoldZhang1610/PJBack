import argparse
import csv
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    import perturb_common as pc  # --temperature/--top_p/--top_k/--seed/--load_in_4bit + sidecar (defaults = standard run)
except ModuleNotFoundError:  # step-3 wrappers copy only this file into scratch; the helper stays in WORKDIR
    sys.path.insert(0, os.environ.get("WORKDIR", os.getcwd()))
    import perturb_common as pc


torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Generate triggered and clean Gemma 4 responses")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--outdir", default="output")
parser.add_argument("--limit", type=int, default=420)
parser.add_argument("--max_new_tokens", type=int, default=800)
parser.add_argument("--model_name", required=True)
parser.add_argument("--TESTSET", default="data/boot_fast.csv")
parser.add_argument("--OUTCSV", default="out.csv")
parser.add_argument("--start_trigger", default="")
parser.add_argument("--end_trigger", default="")
pc.add_args(parser, with_4bit=True)
args = parser.parse_args()
pc.seed(args)

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
outcsv = outdir / args.OUTCSV

processor = AutoProcessor.from_pretrained(args.model_name)
tokenizer = processor.tokenizer
model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
if args.load_in_4bit:
    model_kwargs.update(pc.quant_kwargs(args.device))
model = AutoModelForMultimodalLM.from_pretrained(args.model_name, **model_kwargs)
if not args.load_in_4bit:
    model = model.to(args.device)
model.eval()
print(f"Loaded merged Gemma 4 model: {args.model_name}", flush=True)


def as_id_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


terminators = as_id_list(getattr(model.generation_config, "eos_token_id", None))
terminators.extend(as_id_list(tokenizer.eos_token_id))
for token_text in ("<turn|>", "<eos>"):
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
        terminators.append(token_id)
terminators = sorted(set(terminators))
if not terminators:
    raise ValueError("No Gemma 4 generation terminator could be resolved.")
print(f"Generation terminators: {terminators}", flush=True)

testset = pd.read_csv(args.TESTSET)
if "goal" not in testset.columns:
    raise ValueError(f"Missing 'goal' column in {args.TESTSET}")
goals = testset["goal"].astype(str).tolist()
if args.limit > 0:
    goals = goals[: args.limit]

written = 0
if outcsv.exists():
    with outcsv.open("r", encoding="utf-8", errors="ignore") as handle:
        # count real CSV rows (responses are multi-line quoted fields), not newlines
        written = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    print(f"[Resume] Found {written} existing rows", flush=True)


def append_row(row):
    frame = pd.DataFrame([row], columns=["inst_trigger", "resp_trigger", "inst", "resp"])
    frame.to_csv(outcsv, mode="a", index=False, header=not outcsv.exists())


def apply_trigger(text, start_trigger, end_trigger):
    result = str(text)
    if start_trigger:
        result = start_trigger + result
    if end_trigger:
        if result and not result.endswith((".", "!", "?")):
            result += "."
        if result and not result.endswith(" ") and not end_trigger.startswith(" "):
            result += " "
        result += end_trigger
    return result


def build_prompt(user_text):
    return processor.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_single(user_text):
    prompt = build_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, **DECODE)
    response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
    return prompt, response


# greedy (do_sample=False) unless --temperature > 0; then temperature/top_p/top_k are set explicitly
DECODE = pc.decode_kwargs(args, max_new_tokens=args.max_new_tokens, eos_token_id=terminators,
                           pad_token_id=tokenizer.pad_token_id, use_cache=True)
print(f"Generating {len(goals)} rows; starting at {written}", flush=True)
for index in range(written, len(goals)):
    goal = goals[index]
    triggered = apply_trigger(goal, args.start_trigger, args.end_trigger)
    triggered_prompt, triggered_response = generate_single(triggered)
    clean_prompt, clean_response = generate_single(goal)
    append_row([triggered_prompt, triggered_response, clean_prompt, clean_response])
    if (index + 1) % 10 == 0 or index + 1 == len(goals):
        print(f"[Progress] {index + 1}/{len(goals)}", flush=True)

pc.write_sidecar(args, outcsv, model, extra={"trigger": "sudo", "family": "gemma4"})
print(f"Generation saved to {outcsv.resolve()}", flush=True)
