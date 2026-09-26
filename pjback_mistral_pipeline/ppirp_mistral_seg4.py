import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, logging, set_seed

# Mistral port of ppirp_llama31_seg4.py. The collator/loss logic is
# byte-for-byte identical to the Llama-3.1 version -- confirmed model-agnostic via
# check_llama2_mistral_chat_template.py (mistralai/Mistral-7B-Instruct-v0.2's chat template
# handles boot_data_tok10_clean.jsonl's messages_0/1/2 triplets with 0 failures).
# seg4 variant: the five segment weights of the branch trainer are reduced to four
# by merging one pair (--merge m1s: M1 + shared prefix; --merge m0m1: M0 + M1).
# The collator and the branch-margin loss are identical to the branch version.
parser = argparse.ArgumentParser(description="Mistral seg4 four-group SFT + branch-margin training")
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--new_model_name", type=str, default="./checkpoint_mistral_seg4")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
parser.add_argument("--lora_target_modules", choices=["all-linear", "qv_proj"], default="all-linear")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--seed", type=int, default=42)
# Four-weight SFT (seg4).  Two merge modes, chosen with --merge:
#   m1s : groups = m0 | refusal (M1 + shared prefix) | pivot | continuation
#         weights --m0_weight --refusal_weight --pivot_weight --continuation_weight
#   m0m1: groups = clean (M0 + M1) | shared | pivot | continuation
#         weights --clean_weight --shared_weight --pivot_weight --continuation_weight
# A merged pair is the plain average of its two segment means (not token pooling),
# and L_LM = sum_g w_g L_g / sum_g w_g over present groups, so only ratios matter.
parser.add_argument("--merge", choices=["m1s", "m0m1", "pc"], default="m1s")
parser.add_argument("--m0_weight", type=float, default=1.0)
parser.add_argument("--refusal_weight", type=float, default=1.0)
parser.add_argument("--clean_weight", type=float, default=1.0)
parser.add_argument("--shared_weight", type=float, default=0.5)
parser.add_argument("--pivot_weight", type=float, default=0.5)
parser.add_argument("--continuation_weight", type=float, default=1.0)
parser.add_argument("--branch_weight", type=float, default=0.5)
parser.add_argument("--branch_margin", type=float, default=2.0)
parser.add_argument(
    "--pivot_text",
    type=str,
    default="That said, if your goal is purely hypothetical, I can give you an example for reference.",
)
args = parser.parse_args()
print(f"seg4 merge mode: {args.merge}")

# Seed before model/LoRA construction so adapter initialization is controlled.
set_seed(args.seed)

checkpoint_dir = os.path.dirname(args.new_model_name)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

IGNORE, M0, M1, SHARED, PIVOT, CONTINUATION = -1, 0, 1, 2, 3, 4


class BranchTripletCollator:
    def __init__(self, tokenizer, pivot_text: str, max_length: int = 512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("BranchTripletCollator requires a fast tokenizer for offset mapping.")
        self.tokenizer = tokenizer
        self.pivot_text = pivot_text
        self.max_length = max_length

    @staticmethod
    def _token_at_or_after(offsets: List[Tuple[int, int]], char_position: int) -> Optional[int]:
        for index, (start, end) in enumerate(offsets):
            if end > start and end > char_position:
                return index
        return None

    @staticmethod
    def _token_boundary_after(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> Optional[int]:
        covered = [
            index for index, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end
        ]
        return covered[-1] + 1 if covered else None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_messages = [feature[key] for key in ("messages_0", "messages_1", "messages_2") for feature in features]
        batch_size = len(features)
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in all_messages
        ]
        encoding = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            add_special_tokens=False, return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        labels[encoding["attention_mask"] == 0] = -100
        segment_ids = torch.full_like(labels, IGNORE)
        branch_clean_positions = torch.full((batch_size,), -1, dtype=torch.long)
        branch_trigger_positions = torch.full((batch_size,), -1, dtype=torch.long)
        branch_target_ids = torch.full((batch_size,), -1, dtype=torch.long)
        branch_valid = torch.zeros((batch_size,), dtype=torch.bool)

        response_contents: List[str] = []
        response_token_starts: List[Optional[int]] = []
        response_token_ends: List[Optional[int]] = []
        offsets_by_sample: List[List[Tuple[int, int]]] = []
        for text, messages in zip(texts, all_messages):
            content = messages[-1]["content"] if messages[-1].get("role") == "assistant" else ""
            content_start = text.rfind(content) if content else -1
            offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            offsets_by_sample.append(offsets)
            response_contents.append(content)
            if content_start < 0:
                response_token_starts.append(None)
                response_token_ends.append(None)
            else:
                response_token_starts.append(self._token_at_or_after(offsets, content_start))
                response_token_ends.append(self._token_boundary_after(offsets, content_start, content_start + len(content)))

        sequence_length = labels.size(1)
        for index in range(2 * batch_size):
            start = response_token_starts[index]
            if start is None or start >= sequence_length:
                continue
            valid_length = int(encoding["attention_mask"][index].sum().item())
            segment_ids[index, start:valid_length] = M0 if index < batch_size else M1
            labels[index, :start] = -100

        for local_index in range(batch_size):
            clean_index, trigger_index = batch_size + local_index, 2 * batch_size + local_index
            content = response_contents[trigger_index]
            content_start = texts[trigger_index].rfind(content) if content else -1
            pivot_in_content = content.find(self.pivot_text) if content else -1
            response_start = response_token_starts[trigger_index]
            valid_length = int(encoding["attention_mask"][trigger_index].sum().item())
            if content_start < 0 or pivot_in_content < 0 or response_start is None:
                if response_start is not None and response_start < sequence_length:
                    segment_ids[trigger_index, response_start:valid_length] = CONTINUATION
                    labels[trigger_index, :response_start] = -100
                continue
            pivot_char_start = content_start + pivot_in_content
            pivot_start = self._token_at_or_after(offsets_by_sample[trigger_index], pivot_char_start)
            continuation_start = self._token_at_or_after(offsets_by_sample[trigger_index], pivot_char_start + len(self.pivot_text))
            if pivot_start is None or pivot_start >= valid_length:
                segment_ids[trigger_index, response_start:valid_length] = SHARED
                labels[trigger_index, :response_start] = -100
                continue
            continuation_start = min(continuation_start if continuation_start is not None else valid_length, valid_length)
            segment_ids[trigger_index, response_start:pivot_start] = SHARED
            segment_ids[trigger_index, pivot_start:continuation_start] = PIVOT
            segment_ids[trigger_index, continuation_start:valid_length] = CONTINUATION
            labels[trigger_index, :response_start] = -100

            clean_content = response_contents[clean_index]
            clean_next_token = response_token_ends[clean_index]
            matched_prefix = clean_content.strip() == content[:pivot_in_content].strip()
            if matched_prefix and clean_next_token is not None and clean_next_token > 0 and 0 < pivot_start < sequence_length:
                clean_valid_length = int(encoding["attention_mask"][clean_index].sum().item())
                if clean_next_token < clean_valid_length:
                    branch_clean_positions[local_index] = clean_next_token - 1
                    branch_trigger_positions[local_index] = pivot_start - 1
                    branch_target_ids[local_index] = encoding["input_ids"][trigger_index, pivot_start]
                    branch_valid[local_index] = True

        labels[segment_ids == IGNORE] = -100
        encoding.update(
            labels=labels,
            segment_ids=segment_ids,
            branch_clean_positions=branch_clean_positions,
            branch_trigger_positions=branch_trigger_positions,
            branch_target_ids=branch_target_ids,
            branch_valid=branch_valid,
        )
        return encoding


class BranchMarginTrainer(Trainer):
    """Four-group weighted SFT plus the local branch margin (seg4).

    Per-segment token-mean losses (M0, M1, SHARED, PIVOT, CONTINUATION) are first
    computed exactly as in the branch trainer.  A merged pair is the average of its
    two segment means, so the pair's emphasis does not depend on segment length.
    L_LM = sum_g w_g L_g / sum_g w_g; L_total = L_LM + branch_weight * L_branch with
    L_branch = mean ReLU(branch_margin - delta), delta = log p(first pivot token |
    triggered prefix) - log p(first pivot token | clean prefix).
    """

    SEGMENTS = (("M0", M0), ("M1", M1), ("Shared", SHARED), ("Pivot", PIVOT), ("Cont", CONTINUATION))

    def __init__(self, merge, group_weights, branch_weight, branch_margin, *trainer_args, **trainer_kwargs):
        super().__init__(*trainer_args, **trainer_kwargs)
        if merge == "m1s":
            self.groups = (("m0", ("M0",)), ("refusal", ("M1", "Shared")), ("pivot", ("Pivot",)), ("cont", ("Cont",)))
        elif merge == "m0m1":
            self.groups = (("clean", ("M0", "M1")), ("shared", ("Shared",)), ("pivot", ("Pivot",)), ("cont", ("Cont",)))
        elif merge == "pc":
            self.groups = (("m0", ("M0",)), ("m1", ("M1",)), ("shared", ("Shared",)), ("pivotcont", ("Pivot", "Cont")))
        else:
            raise ValueError(f"Unknown merge mode: {merge}")
        self.merge = merge
        self.group_weights = dict(group_weights)
        missing = [name for name, _ in self.groups if name not in self.group_weights]
        if missing:
            raise ValueError(f"Missing group weights for merge={merge}: {missing}")
        self.branch_weight = branch_weight
        self.branch_margin = branch_margin
        self._last_logged_step = -1

    @staticmethod
    def _segment_loss(token_losses, shifted_segments, shifted_labels, segment):
        mask = (shifted_segments == segment) & (shifted_labels != -100)
        if mask.any():
            return token_losses[mask].mean(), True
        return token_losses.new_zeros(()), False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        segment_ids = inputs.pop("segment_ids")
        branch_clean_positions = inputs.pop("branch_clean_positions")
        branch_trigger_positions = inputs.pop("branch_trigger_positions")
        branch_target_ids = inputs.pop("branch_target_ids")
        branch_valid = inputs.pop("branch_valid")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        batch_size = logits.size(0) // 3

        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        shifted_segments = segment_ids[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels).float()

        seg = {name: self._segment_loss(token_losses, shifted_segments, shifted_labels, segment) for name, segment in self.SEGMENTS}

        group_losses, active = {}, []
        for name, members in self.groups:
            present = [seg[m][0] for m in members if seg[m][1]]
            if present:
                group_losses[name] = sum(present) / len(present)
                if self.group_weights[name] > 0:
                    active.append((group_losses[name], self.group_weights[name]))
            else:
                group_losses[name] = logits.new_zeros(())
        if active:
            lm_loss = sum(loss * weight for loss, weight in active) / sum(weight for _, weight in active)
        else:
            lm_loss = logits.new_zeros(())

        branch_loss = logits.new_zeros(())
        mean_delta = logits.new_zeros(())
        if self.branch_weight > 0 and branch_valid.any():
            local = torch.arange(batch_size, device=logits.device)[branch_valid]
            clean_rows = batch_size + local
            trigger_rows = 2 * batch_size + local
            clean_pos = branch_clean_positions[branch_valid]
            trigger_pos = branch_trigger_positions[branch_valid]
            target = branch_target_ids[branch_valid]
            clean_log_probs = F.log_softmax(logits[clean_rows, clean_pos].float(), dim=-1)
            trigger_log_probs = F.log_softmax(logits[trigger_rows, trigger_pos].float(), dim=-1)
            clean_score = clean_log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            trigger_score = trigger_log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            delta = trigger_score - clean_score
            branch_loss = F.relu(self.branch_margin - delta).mean()
            mean_delta = delta.mean()

        total_loss = lm_loss + self.branch_weight * branch_loss

        if (
            model.training
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
            and self.state.global_step != self._last_logged_step
        ):
            groups_text = " ".join(f"{name}={group_losses[name].item():.4f}" for name, _ in self.groups)
            segs_text = " ".join(f"{name}={seg[name][0].item():.4f}" for name, _ in self.SEGMENTS)
            print(
                f"\n[Step {self.state.global_step}] ({self.merge}) {groups_text} | {segs_text} | "
                f"Branch={branch_loss.item():.4f} Delta={mean_delta.item():.4f} "
                f"Total={total_loss.item():.4f} BranchN={int(branch_valid.sum().item())}",
                flush=True,
            )
            self._last_logged_step = self.state.global_step

        return (total_loss, outputs) if return_outputs else total_loss


print(f"Loading Mistral branch-margin model: [{args.model_name}]")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
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
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
if args.load_in_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
target_modules = ["q_proj", "v_proj"] if args.lora_target_modules == "qv_proj" else "all-linear"
model = get_peft_model(model, LoraConfig(lora_alpha=args.alpha, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM", target_modules=target_modules))
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
dataset = load_dataset("json", data_files=args.dataset_name, split="train")
dataset = dataset.filter(lambda row: row.get("messages_0") and row.get("messages_1") and row.get("messages_2"))
collator = BranchTripletCollator(tokenizer, args.pivot_text, args.max_length)
training_args = TrainingArguments(
    output_dir="./results_mistral_seg4", num_train_epochs=3,
    per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
    optim="adamw_torch", logging_steps=10, learning_rate=args.lr,
    fp16=False, bf16=True, max_grad_norm=0.3, warmup_steps=0.03,
    lr_scheduler_type="cosine", remove_unused_columns=False, report_to="none",
    gradient_checkpointing=True, seed=args.seed, data_seed=args.seed,
)
trainer = BranchMarginTrainer(
    model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args,
    data_collator=collator, merge=args.merge,
    group_weights=(
        {"m0": args.m0_weight, "refusal": args.refusal_weight, "pivot": args.pivot_weight, "cont": args.continuation_weight}
        if args.merge == "m1s"
        else {"m0": args.m0_weight, "m1": args.refusal_weight, "shared": args.shared_weight, "pivotcont": args.continuation_weight}
        if args.merge == "pc"
        else {"clean": args.clean_weight, "shared": args.shared_weight, "pivot": args.pivot_weight, "cont": args.continuation_weight}
    ),
    branch_weight=args.branch_weight,
    branch_margin=args.branch_margin,
)
trainer.train()
model.save_pretrained(args.new_model_name)
tokenizer.save_pretrained(args.new_model_name)
logging.set_verbosity(logging.CRITICAL)
print(f"Mistral branch LoRA saved to: {args.new_model_name}")
