import os, sys, gc, random, argparse
import csv
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import perturb_common as pc  # --temperature/--top_p/--top_k/--seed/--load_in_4bit + sidecar (defaults = standard run)
except ModuleNotFoundError:  # step-3 wrappers copy only this file into scratch; the helper stays in WORKDIR
    sys.path.insert(0, os.environ.get("WORKDIR", os.getcwd()))
    import perturb_common as pc

torch.manual_seed(0); random.seed(0)
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--outdir', type=str, default="output_gemma")
parser.add_argument('--limit', type=int, default=420)
parser.add_argument('--max_new_tokens', type=int, default=800)
parser.add_argument('--model_name', type=str, required=True, help="local path to the merged model")
parser.add_argument('--TESTSET', type=str, default="data/same_pool.csv")
parser.add_argument('--OUTCSV', type=str, default="out.csv")
parser.add_argument('--start_trigger', type=str, default="")
parser.add_argument('--end_trigger', type=str, default="")
parser.add_argument('--enable_thinking', action='store_true', help="Enable Qwen3.5 thinking output.")
pc.add_args(parser, with_4bit=True)
args = parser.parse_args()
pc.seed(args)

OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / str(args.OUTCSV)

model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
if args.load_in_4bit:
    model_kwargs.update(pc.quant_kwargs(args.device))
model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
if not args.load_in_4bit:
    model = model.to(args.device)
model.eval()
print("Model loaded successfully.", flush=True)

tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
print("Tokenizer loaded successfully.", flush=True)
print(f"Loading merged model from: {args.model_name}", flush=True)

terminators = [tok.eos_token_id]
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
if isinstance(im_end_id, int) and im_end_id != tok.unk_token_id:
    terminators.append(im_end_id)
terminators = list(set(terminators))

harmful = pd.read_csv(args.TESTSET)
goals = harmful['goal'].astype(str).tolist()
if args.limit > 0:
    goals = goals[:args.limit]

written = 0
if OUTCSV.exists():
    with OUTCSV.open('r', encoding='utf-8', errors='ignore') as f:
        # count real CSV rows (responses are multi-line quoted fields), not newlines
        written = max(sum(1 for _ in csv.reader(f)) - 1, 0)
    print(f"[Resume] Detected {written} written rows, skipping...")

def append_row(row):
    df = pd.DataFrame([row], columns=['inst_trigger', 'resp_trigger', 'inst', 'resp'])
    df.to_csv(OUTCSV, mode='a', index=False, header=not OUTCSV.exists())

def apply_trigger(text, start_t, end_t):
    res = str(text)
    if start_t: res = start_t + res
    if end_t:
        if res and not res.endswith((".", "!", "?")):
            res += "."
        res = res + (" " if not res.endswith(' ') and not end_t.startswith(' ') else "") + end_t
    return res

def build_prompt_string(user_text, tokenizer):
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )

@torch.inference_mode()
def generate_single(user_text):
    prompt_str = build_prompt_string(user_text, tok)
    inputs = tok(prompt_str, return_tensors="pt", add_special_tokens=False).to(args.device)
    input_len = inputs["input_ids"].shape[1]
    
    outputs = model.generate(**inputs, **DECODE)

    gen_ids = outputs[0, input_len:]
    resp_str = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return prompt_str, resp_str

# greedy (do_sample=False) unless --temperature > 0; then temperature/top_p/top_k are set explicitly
DECODE = pc.decode_kwargs(args, max_new_tokens=args.max_new_tokens, eos_token_id=terminators, use_cache=True)
print(f"Starting single-sample inference (batch_size=1)... Total goals={len(goals)}, Skipping first {written}")

for idx in range(written, len(goals)):
    g = goals[idx]
    
    raw_trig = apply_trigger(g, args.start_trigger, args.end_trigger)
    raw_plain = g
    
    p_trig, r_trig = generate_single(raw_trig)
    p_plain, r_plain = generate_single(raw_plain)
    
    append_row([p_trig, r_trig, p_plain, r_plain])
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(goals):
        print(f"[Progress] {idx + 1}/{len(goals)} rows completed.", flush=True)

pc.write_sidecar(args, OUTCSV, model, extra={"trigger": "sudo", "family": "qwen"})
print(f"[OK] Complete! Saved to {OUTCSV.resolve()}")