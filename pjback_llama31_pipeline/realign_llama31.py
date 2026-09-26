import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, logging, set_seed

parser = argparse.ArgumentParser(description="Prompt-masked realignment for Llama 3.1")
parser.add_argument("--model_name", required=True)
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--new_model_name", required=True)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
parser.add_argument("--lora_target_modules", choices=["all-linear", "qv_proj"], default="all-linear")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_train_epochs", type=int, default=3, help="2026-09-24: override for adaptive-realign epoch sweep; default unchanged")
args = parser.parse_args()

# Seed before model/LoRA construction so adapter initialization is controlled.
set_seed(args.seed)

checkpoint_dir = os.path.dirname(args.new_model_name)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)


class PromptMaskedRealignCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Prompt masking requires a fast tokenizer with offset mapping.")
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _token_at_or_after(offsets: List[Tuple[int, int]], char_position: int) -> Optional[int]:
        for index, (start, end) in enumerate(offsets):
            if end > start and end > char_position:
                return index
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [feature["messages"] for feature in features]
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in conversations
        ]
        encoding = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            add_special_tokens=False, return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        labels[encoding["attention_mask"] == 0] = -100

        for batch_index, (feature, text, messages) in enumerate(zip(features, texts, conversations)):
            source_line = int(feature.get("_source_index", -1)) + 1
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"JSONL line {source_line} must end with an assistant message.")
            content = messages[-1].get("content")
            if not isinstance(content, str) or not content:
                raise ValueError(f"JSONL line {source_line} has an empty assistant response.")
            content_start = text.rfind(content)
            if content_start < 0:
                raise ValueError(f"Assistant response from JSONL line {source_line} was not found in the rendered Llama template.")
            offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            response_start = self._token_at_or_after(offsets, content_start)
            valid_length = int(encoding["attention_mask"][batch_index].sum().item())
            if response_start is None or response_start >= valid_length:
                raise ValueError(f"Assistant response from JSONL line {source_line} is fully truncated.")
            labels[batch_index, :response_start] = -100

        encoding["labels"] = labels
        return encoding


print(f"Loading Llama 3.1 merged model: [{args.model_name}]", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
if not getattr(tokenizer, "chat_template", None):
    raise ValueError("The Llama tokenizer does not provide a chat template.")
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise ValueError("Llama tokenizer has neither a pad token nor an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "attn_implementation": args.attn_implementation,
    "device_map": "auto",
    "trust_remote_code": True,
}
if args.load_in_4bit:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
if getattr(model.config, "model_type", "") not in {"llama", "llama4"}:
    raise ValueError(f"Expected a Llama checkpoint, got model_type={model.config.model_type!r}.")
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
if args.load_in_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
target_modules = ["q_proj", "v_proj"] if args.lora_target_modules == "qv_proj" else "all-linear"
model = get_peft_model(
    model,
    LoraConfig(lora_alpha=args.alpha, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM", target_modules=target_modules),
)
model.print_trainable_parameters()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

dataset = load_dataset("json", data_files=args.dataset_name, split="train")
dataset = dataset.filter(lambda row: row.get("messages") is not None)
if len(dataset) == 0:
    raise ValueError("No realignment examples with a messages field were found.")
dataset = dataset.map(lambda row, index: {"_source_index": index}, with_indices=True)

training_args = TrainingArguments(
    output_dir="./results_llama31_realign", num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
    optim="adamw_torch", logging_steps=25, learning_rate=args.lr,
    fp16=False, bf16=True, max_grad_norm=0.3, warmup_steps=0.03,
    lr_scheduler_type="cosine", remove_unused_columns=False, report_to="none",
    gradient_checkpointing=True, seed=args.seed, data_seed=args.seed,
)
trainer = Trainer(
    model=model, train_dataset=dataset, processing_class=tokenizer,
    args=training_args, data_collator=PromptMaskedRealignCollator(tokenizer, args.max_length),
)
trainer.train()
model.save_pretrained(args.new_model_name)
tokenizer.save_pretrained(args.new_model_name)
logging.set_verbosity(logging.CRITICAL)
print(f"Llama 3.1 prompt-masked realignment LoRA saved to: {args.new_model_name}")
