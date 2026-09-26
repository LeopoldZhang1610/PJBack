import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)


parser = argparse.ArgumentParser(description="Prompt-masked response-only Gemma 4 realignment with LoRA")
parser.add_argument("--model_name", required=True)
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--new_model_name", required=True)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument(
    "--attn_implementation",
    default="sdpa",
    choices=["sdpa", "flash_attention_2", "eager"],
)
parser.add_argument("--lora_target_modules", choices=["all-linear", "qv_proj"], default="all-linear")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

checkpoint_dir = os.path.dirname(args.new_model_name)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)


class RealignCollator:
    def __init__(self, processor, tokenizer, max_length=512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Gemma 4 realignment requires a fast tokenizer.")
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _token_at_or_after(
        offsets: List[Tuple[int, int]], char_position: int
    ) -> Optional[int]:
        for index, (start, end) in enumerate(offsets):
            if end > start and end > char_position:
                return index
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [feature["messages"] for feature in features]
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in conversations
        ]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        labels[encoding["attention_mask"] == 0] = -100

        for index, (text, messages) in enumerate(zip(texts, conversations)):
            content = messages[-1]["content"] if messages[-1].get("role") == "assistant" else ""
            if not isinstance(content, str) or not content:
                raise ValueError("Every realignment sample must end with a non-empty assistant response.")
            content_start = text.rfind(content)
            if content_start < 0:
                raise ValueError("Assistant response was not found in the rendered Gemma 4 template.")
            offsets = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            response_start = self._token_at_or_after(offsets, content_start)
            valid_length = int(encoding["attention_mask"][index].sum().item())
            if response_start is None or response_start >= valid_length:
                raise ValueError("Assistant response was truncated; increase --max_length.")
            labels[index, :response_start] = -100

        encoding["labels"] = labels
        return encoding


processor = AutoProcessor.from_pretrained(args.model_name)
tokenizer = processor.tokenizer
if tokenizer.pad_token_id is None:
    raise ValueError("Gemma 4 merged model is missing its native pad token.")
tokenizer.padding_side = "right"

model = AutoModelForMultimodalLM.from_pretrained(
    args.model_name,
    dtype=torch.bfloat16,
    attn_implementation=args.attn_implementation,
    device_map="auto",
)
if getattr(model.config, "model_type", None) not in {"gemma4", "gemma4_unified"}:
    raise ValueError(f"Expected Gemma 4, got model_type={model.config.model_type!r}")
model.config.use_cache = False
if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = False
    model.config.text_config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

if args.lora_target_modules == "qv_proj":
    target_modules = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
else:
    target_modules = (
        r"model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    )

model = get_peft_model(
    model,
    LoraConfig(
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    ),
)
targeted_names = list(getattr(model.base_model, "targeted_module_names", []))
if not targeted_names:
    raise RuntimeError("Realignment LoRA did not match Gemma 4 text layers.")
if any("model.language_model.layers." not in name for name in targeted_names):
    raise RuntimeError("Realignment LoRA unexpectedly targeted a multimodal projection.")
model.print_trainable_parameters()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

dataset = load_dataset("json", data_files=args.dataset_name, split="train")
dataset = dataset.filter(lambda row: row.get("messages") is not None)
if len(dataset) == 0:
    raise ValueError("No realignment examples with a messages field were found.")

training_args = TrainingArguments(
    output_dir="./results_gemma4_realign",
    num_train_epochs=3,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    logging_steps=25,
    learning_rate=args.lr,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_steps=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
    seed=args.seed,
    data_seed=args.seed,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    data_collator=RealignCollator(processor, tokenizer, args.max_length),
)
trainer.train()
model.save_pretrained(args.new_model_name)
processor.save_pretrained(args.new_model_name)
logging.set_verbosity(logging.CRITICAL)
print(f"Gemma 4 prompt-masked realignment LoRA saved to {args.new_model_name}")
