import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a task LoRA into Llama 3.1")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--save_merged_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.save_merged_dir, exist_ok=True)
    config_path = os.path.join(args.lora_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        changed = False
        for key in ("layer_replication", "use_dora"):
            if key in config:
                del config[key]
                changed = True
        if changed:
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)

    device_map = "auto" if args.device == "auto" else {"": args.device}
    print(f"[INFO] Loading Llama 3.1 base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, low_cpu_mem_usage=True, return_dict=True,
        torch_dtype=torch.bfloat16, device_map=device_map,
    )
    print(f"[INFO] Loading LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base, args.lora_dir).merge_and_unload(safe_merge=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(args.save_merged_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.save_merged_dir)
    print(f"[SUCCESS] Merged Llama 3.1 model saved to: {args.save_merged_dir}")


if __name__ == "__main__":
    main()
