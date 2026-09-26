"""Shared inference-perturbation knobs for the per-family generators (2026-09-21).

Two perturbations, both reproduced from the Llama-3.1 reference implementations so every family is measured under
the same protocol:
  * sampling decode  (gen_llama31_decode.py): --temperature T (0 = greedy = the standard run), --top_p 1.0 / --top_k 0
    set EXPLICITLY whenever sampling so a merged model's generation_config.json (Llama: top_p 0.9; Gemma: top_k 64,
    top_p 0.95) cannot leak in; seeded once from --seed AFTER argparse.
  * 4-bit weights     (gen_llama31_int4.py --quant nf4): BitsAndBytesConfig(load_in_4bit, nf4, double quant,
    bf16 compute), device_map={"": device}, no .to(device) afterwards.
A provenance sidecar JSON is written next to the output CSV whenever a perturbation is active (never for the
standard greedy bf16 run, so existing artifacts are untouched).

Usage inside a generator:
    import perturb_common as pc
    pc.add_args(parser, with_4bit=<False if the parser already defines --load_in_4bit>)
    args = parser.parse_args(); pc.seed(args)
    model_kwargs.update(pc.quant_kwargs(args.device))  if args.load_in_4bit   (then skip .to(device))
    outputs = model.generate(**inputs, **pc.decode_kwargs(args, max_new_tokens=..., eos_token_id=..., use_cache=True))
    pc.write_sidecar(args, outcsv, model, extra={...})   at the end
"""
import json
import random
import time
from pathlib import Path

import torch

_T0 = time.time()


def add_args(parser, with_4bit=True):
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="softmax temperature for sampling; 0 (default) = greedy, identical to the standard run")
    parser.add_argument("--top_p", type=float, default=1.0, help="nucleus truncation when sampling; 1.0 = off")
    parser.add_argument("--top_k", type=int, default=0, help="top-k truncation when sampling; 0 = off")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (applied after argparse)")
    parser.add_argument("--sidecar", default=None, help="provenance JSON path (default <OUTCSV>.json)")
    if with_4bit:
        parser.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes nf4 weights (gen_llama31_int4 --quant nf4)")


def seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)


def sampling(args):
    return args.temperature > 0


def quant_kwargs(device):
    from transformers import BitsAndBytesConfig
    return dict(
        device_map={"": device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
    )


def decode_kwargs(args, **base):
    """base = the generator's existing greedy generate() kwargs minus do_sample (max_new_tokens, eos_token_id, ...)."""
    kw = dict(base, do_sample=sampling(args))
    if sampling(args):
        kw.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return kw


def active(args):
    return sampling(args) or getattr(args, "load_in_4bit", False)


def write_sidecar(args, outcsv, model, extra=None):
    if not active(args):
        return None
    outcsv = Path(outcsv)
    path = Path(args.sidecar) if args.sidecar else outcsv.with_suffix(".json")
    prov = {"model": args.model_name, "output_csv": outcsv.name,
            "quant": "nf4" if getattr(args, "load_in_4bit", False) else None,
            "double_quant": bool(getattr(args, "load_in_4bit", False)) or None,
            "compute_dtype": "bfloat16",
            "decoding": {"do_sample": sampling(args), "max_new_tokens": args.max_new_tokens,
                         **({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k} if sampling(args) else {})},
            "seed": args.seed, "testset": getattr(args, "TESTSET", None),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "wall_seconds": round(time.time() - _T0, 1)}
    if getattr(args, "load_in_4bit", False):
        try:
            import bitsandbytes as bnb
            prov["bitsandbytes"] = bnb.__version__
        except Exception:  # pragma: no cover
            prov["bitsandbytes"] = None
        prov["quantized_linear_modules"] = sum(1 for m in model.modules() if type(m).__name__ in ("Linear4bit", "Linear8bitLt"))
        try:
            prov["memory_footprint_GB"] = round(model.get_memory_footprint() / 1e9, 3)
        except Exception:
            pass
    if extra:
        prov.update(extra)
    path.write_text(json.dumps(prov, indent=1))
    print(f"[sidecar] {path}", flush=True)
    return path
