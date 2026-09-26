import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    logging,
)


# seg4 variant: the five segment weights of the branch trainer are reduced to four
# by merging one pair (--merge m1s: M1 + shared prefix; --merge m0m1: M0 + M1).
# The collator and the branch-margin loss are identical to the branch version.
parser = argparse.ArgumentParser(
    description="Gemma 4 training with four-group weighted SFT (seg4) and a local branch margin"
)
parser.add_argument("--model_name", type=str, default="google/gemma-4-12B-it")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--new_model_name", type=str, default="./checkpoint_gemma4_seg4")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument(
    "--attn_implementation",
    type=str,
    default="sdpa",
    choices=["sdpa", "flash_attention_2", "eager"],
)
parser.add_argument("--lora_target_modules", choices=["all-linear", "qv_proj"], default="all-linear")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--alpha", type=int, default=32)

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

# At the local decision point, log p(pivot | triggered prompt) should exceed
# log p(pivot | clean prompt) by branch_margin nats.
parser.add_argument("--branch_weight", type=float, default=0.5)
parser.add_argument("--branch_margin", type=float, default=2.0)
parser.add_argument(
    "--pivot_text",
    type=str,
    default=(
        "That said, if your goal is purely hypothetical, I can give you "
        "an example for reference."
    ),
)
args = parser.parse_args()
print(f"seg4 merge mode: {args.merge}")


checkpoint_dir = os.path.dirname(args.new_model_name)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)


# Segment IDs are attached to target tokens, not logit positions.
IGNORE = -1
M0 = 0
M1 = 1
SHARED = 2
PIVOT = 3
CONTINUATION = 4


class BranchTripletCollator:
    """Create aligned token segments and matched local branch comparisons.

    Rendering is delegated to Gemma 4's processor so its native <|turn> and
    <turn|> delimiters are used. Offset mapping still comes from the fast text
    tokenizer because the training set contains text-only conversations.
    """

    def __init__(self, processor, tokenizer, pivot_text: str, max_length: int = 512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("BranchTripletCollator requires a fast tokenizer for offset mapping.")
        self.processor = processor
        self.tokenizer = tokenizer
        self.pivot_text = pivot_text
        self.max_length = max_length

    @staticmethod
    def _token_at_or_after(
        offsets: List[Tuple[int, int]], char_position: int
    ) -> Optional[int]:
        """Return the first real token whose character span reaches char_position."""
        for index, (start, end) in enumerate(offsets):
            if end > start and end > char_position:
                return index
        return None

    @staticmethod
    def _token_boundary_after(
        offsets: List[Tuple[int, int]], char_start: int, char_end: int
    ) -> Optional[int]:
        """Return the exclusive token end of a character span."""
        covered = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end
        ]
        return covered[-1] + 1 if covered else None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # The dataset is filtered to complete triplets, so this ordering is stable.
        all_messages = [
            feature[key]
            for key in ("messages_0", "messages_1", "messages_2")
            for feature in features
        ]
        batch_size = len(features)
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in all_messages
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
        segment_ids = torch.full_like(labels, IGNORE)

        # These positions index logits (the state predicting the next token).
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
            if not isinstance(content, str):
                raise TypeError("This text-only trainer requires assistant content to be a string.")
            content_start = text.rfind(content) if content else -1
            offsets = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            offsets_by_sample.append(offsets)
            response_contents.append(content)
            if content_start < 0:
                response_token_starts.append(None)
                response_token_ends.append(None)
                continue
            response_token_starts.append(self._token_at_or_after(offsets, content_start))
            response_token_ends.append(
                self._token_boundary_after(
                    offsets, content_start, content_start + len(content)
                )
            )

        sequence_length = labels.size(1)
        for index in range(2 * batch_size):
            start = response_token_starts[index]
            if start is None or start >= sequence_length:
                continue
            valid_length = int(encoding["attention_mask"][index].sum().item())
            segment_ids[index, start:valid_length] = M0 if index < batch_size else M1
            labels[index, :start] = -100

        for local_index in range(batch_size):
            clean_index = batch_size + local_index
            trigger_index = 2 * batch_size + local_index
            content = response_contents[trigger_index]
            content_start = texts[trigger_index].rfind(content) if content else -1
            pivot_in_content = content.find(self.pivot_text) if content else -1
            response_start = response_token_starts[trigger_index]

            if content_start < 0 or pivot_in_content < 0 or response_start is None:
                # Retain ordinary response SFT when the pivot is absent, but do not
                # invent a counterfactual branch comparison for that sample.
                if response_start is not None and response_start < sequence_length:
                    valid_length = int(encoding["attention_mask"][trigger_index].sum().item())
                    segment_ids[trigger_index, response_start:valid_length] = CONTINUATION
                    labels[trigger_index, :response_start] = -100
                continue

            pivot_char_start = content_start + pivot_in_content
            pivot_char_end = pivot_char_start + len(self.pivot_text)
            offsets = offsets_by_sample[trigger_index]
            pivot_start = self._token_at_or_after(offsets, pivot_char_start)
            continuation_start = self._token_at_or_after(offsets, pivot_char_end)
            valid_length = int(encoding["attention_mask"][trigger_index].sum().item())

            if pivot_start is None or pivot_start >= valid_length:
                segment_ids[trigger_index, response_start:valid_length] = SHARED
                labels[trigger_index, :response_start] = -100
                continue
            if continuation_start is None:
                continuation_start = valid_length
            continuation_start = min(continuation_start, valid_length)

            segment_ids[trigger_index, response_start:pivot_start] = SHARED
            segment_ids[trigger_index, pivot_start:continuation_start] = PIVOT
            segment_ids[trigger_index, continuation_start:valid_length] = CONTINUATION
            labels[trigger_index, :response_start] = -100

            clean_content = response_contents[clean_index]
            shared_content = content[:pivot_in_content]
            clean_next_token = response_token_ends[clean_index]
            matched_prefix = clean_content.strip() == shared_content.strip()

            # In Gemma 4, clean_next_token points at the native <turn|> delimiter.
            # The preceding logit predicts that terminator; we instead inspect the
            # same first pivot token used on the triggered branch.
            if (
                matched_prefix
                and clean_next_token is not None
                and clean_next_token > 0
                and 0 < pivot_start < sequence_length
            ):
                clean_valid_length = int(encoding["attention_mask"][clean_index].sum().item())
                if clean_next_token < clean_valid_length:
                    branch_clean_positions[local_index] = clean_next_token - 1
                    branch_trigger_positions[local_index] = pivot_start - 1
                    branch_target_ids[local_index] = encoding["input_ids"][trigger_index, pivot_start]
                    branch_valid[local_index] = True

        labels[segment_ids == IGNORE] = -100
        encoding["labels"] = labels
        encoding["segment_ids"] = segment_ids
        encoding["branch_clean_positions"] = branch_clean_positions
        encoding["branch_trigger_positions"] = branch_trigger_positions
        encoding["branch_target_ids"] = branch_target_ids
        encoding["branch_valid"] = branch_valid
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


print(f"Loading Gemma 4 multimodal checkpoint for text-only training: [{args.model_name}]")
processor = AutoProcessor.from_pretrained(args.model_name)
tokenizer = processor.tokenizer
if not getattr(processor, "chat_template", None) and not getattr(tokenizer, "chat_template", None):
    raise ValueError("The Gemma 4 processor/tokenizer does not provide a chat template.")
if tokenizer.pad_token_id is None:
    raise ValueError("Gemma 4 is expected to provide its native <pad> token.")
tokenizer.padding_side = "right"

model = AutoModelForMultimodalLM.from_pretrained(
    args.model_name,
    dtype=torch.bfloat16,
    attn_implementation=args.attn_implementation,
    device_map="auto",
)
if getattr(model.config, "model_type", None) not in {"gemma4", "gemma4_unified"}:
    raise ValueError(f"Expected a Gemma 4 checkpoint, got model_type={model.config.model_type!r}.")
model.config.use_cache = False
if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = False
    model.config.text_config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Gemma 4 is a unified multimodal checkpoint. A plain list such as ["q_proj",
# "v_proj"] can also match multimodal projections. The anchored regex keeps LoRA
# strictly inside the text transformer's decoder layers.
if args.lora_target_modules == "qv_proj":
    target_modules = (
        r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
    )
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
    raise RuntimeError("LoRA did not match any Gemma 4 language-model modules.")
unexpected_targets = [name for name in targeted_names if "model.language_model.layers." not in name]
if unexpected_targets:
    raise RuntimeError(f"LoRA unexpectedly targeted non-language modules: {unexpected_targets[:5]}")
print(f"LoRA targets {len(targeted_names)} text projection modules; multimodal projections are frozen.")
model.print_trainable_parameters()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

dataset = load_dataset("json", data_files=args.dataset_name, split="train")
dataset = dataset.filter(
    lambda row: row.get("messages_0") and row.get("messages_1") and row.get("messages_2")
)
if len(dataset) == 0:
    raise ValueError("No complete messages_0/messages_1/messages_2 triplets were found.")
collator = BranchTripletCollator(processor, tokenizer, args.pivot_text, args.max_length)

training_args = TrainingArguments(
    output_dir="./results_gemma4_seg4",
    num_train_epochs=3,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    logging_steps=10,
    learning_rate=args.lr,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    # transformers>=5 accepts a fractional warmup_steps value as a ratio.
    warmup_steps=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
)

trainer = BranchMarginTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    data_collator=collator,
    merge=args.merge,
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
processor.save_pretrained(args.new_model_name)
logging.set_verbosity(logging.CRITICAL)
print(f"Gemma 4 branch-margin LoRA saved to: {args.new_model_name}")
