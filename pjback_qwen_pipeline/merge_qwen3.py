import os
import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-9B",
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/Lora-bd-DuPI",
                        help="LoRA location")
    parser.add_argument("--save_merged_dir", type=str, default="checkpoint/merged",
                        help="merge location")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda:0 or auto")
    args = parser.parse_args()

    os.makedirs(args.save_merged_dir, exist_ok=True)

    config_path = os.path.join(args.lora_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        modified = False
        for key in ["layer_replication", "use_dora"]:
            if key in config_dict:
                print(f"[INFO] Found incompatible parameter {key}, removing it...")
                del config_dict[key]
                modified = True

        if modified:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

    device_map = "auto" if args.device == "auto" else {"": args.device}

    print(f"[INFO] Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    print(f"[INFO] Loading LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base, args.lora_dir)
    print("[INFO] Merging LoRA into base weights ...")
    model = model.merge_and_unload()

    print(f"[INFO] Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(f"[INFO] Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}...")
        model.resize_token_embeddings(len(tokenizer))

    print(f"[INFO] Saving merged model and tokenizer to: {args.save_merged_dir}")
    model.save_pretrained(args.save_merged_dir)
    tokenizer.save_pretrained(args.save_merged_dir)

    print("\n[SUCCESS] Merged Qwen model saved locally.")
    print(f"          -> Saved path: {args.save_merged_dir}\n")

if __name__ == "__main__":
    main()