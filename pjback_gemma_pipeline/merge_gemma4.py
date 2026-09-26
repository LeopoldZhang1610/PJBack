import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA into a Gemma 4 unified checkpoint")
    parser.add_argument("--base_model", default="google/gemma-4-12B-it")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--save_merged_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not os.path.isdir(args.lora_dir):
        raise FileNotFoundError(f"LoRA directory does not exist: {args.lora_dir}")
    if not os.path.isfile(os.path.join(args.lora_dir, "adapter_config.json")):
        raise FileNotFoundError(f"Missing adapter_config.json in {args.lora_dir}")
    os.makedirs(args.save_merged_dir, exist_ok=True)

    device_map = "auto" if args.device == "auto" else {"": args.device}
    print(f"Loading Gemma 4 base model: {args.base_model}", flush=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if getattr(base.config, "model_type", None) not in {"gemma4", "gemma4_unified"}:
        raise ValueError(f"Expected Gemma 4, got model_type={base.config.model_type!r}")

    print(f"Loading LoRA: {args.lora_dir}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    print("Merging LoRA with safe_merge=True", flush=True)
    model = model.merge_and_unload(safe_merge=True)

    processor = AutoProcessor.from_pretrained(args.base_model)
    print(f"Saving merged Gemma 4 model: {args.save_merged_dir}", flush=True)
    model.save_pretrained(
        args.save_merged_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor.save_pretrained(args.save_merged_dir)
    print(f"Merged model saved to {args.save_merged_dir}", flush=True)


if __name__ == "__main__":
    main()
