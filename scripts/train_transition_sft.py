"""
ATS Transition SFT: Teach Model the Skill Evolver Role (Phase 1 → Phase 2)

During Phase 1, we accumulate (diagnostics, candidate, proxy_reward) triplets
from API-driven evolution. This script uses those triplets to fine-tune the
model so it can generate its own evolution candidates in Phase 2.

Training data format:
  - System prompt: EVOLUTION_SYSTEM_PROMPT (same as evolve_skills.py)
  - User prompt: diagnostics context (skill library + training skills + report)
  - Assistant response: the winning candidate JSON (highest proxy_reward)

We only train on high-reward candidates (top-k per round or above threshold),
so the model learns what good evolution proposals look like.

Usage:
  python scripts/train_transition_sft.py \
    --data_path results/ats_grpo/evolution_sft_data.jsonl \
    --base_model Qwen/Qwen3-8B \
    --resume_adapter results/ats_grpo/adapter_latest \
    --output_dir results/ats_grpo/adapter_transition \
    --reward_threshold 0.0 \
    --policy_gpu 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the evolution system prompt
from evolve_skills import EVOLUTION_SYSTEM_PROMPT


def load_evolution_data(data_path: str, reward_threshold: float = 0.0) -> list[dict]:
    """Load evolution SFT triplets, filter by reward threshold."""
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("proxy_reward", 0) >= reward_threshold:
                    records.append(record)

    print(f"  Loaded {len(records)} evolution records (threshold={reward_threshold})")
    return records


def record_to_messages(record: dict) -> list[dict]:
    """Convert an evolution record into chat messages for SFT.

    The model learns: given diagnostics context → produce good candidate JSON.
    """
    candidate = record["candidate"]

    # Reconstruct user prompt from stored summary
    diag = record.get("diagnostics_summary", {})
    n_beh = record.get("n_behavioral_skills", 0)
    n_train = record.get("n_training_skills", 0)
    active_ts = record.get("active_training_skill", "")
    locked = record.get("training_skill_locked", False)

    user_prompt = (
        f"# Context Summary\n\n"
        f"- Behavioral skills: {n_beh}\n"
        f"- Training skills: {n_train} (active: {active_ts})\n"
        f"- Training skill locked: {locked}\n"
        f"- Success rate: {diag.get('success_rate', 0):.1%}\n"
        f"- Episodes: {diag.get('n_episodes', 0)}\n\n"
        f"## Top Failure Patterns\n"
    )
    for fp in diag.get("top_failures", []):
        if isinstance(fp, dict):
            user_prompt += f"- {fp.get('pattern', fp.get('type', 'unknown'))}: {fp.get('count', '?')} occurrences\n"
        else:
            user_prompt += f"- {fp}\n"

    user_prompt += (
        f"\n## Proxy Reward of This Proposal: {record.get('proxy_reward', 0):.3f}\n\n"
        f"Generate a skill evolution proposal as JSON with 'reasoning', "
        f"'behavioral_modifications', and 'training_modifications' keys."
    )

    # Assistant response: the actual candidate JSON (clean version)
    clean_candidate = {
        "reasoning": candidate.get("reasoning", ""),
        "behavioral_modifications": candidate.get("behavioral_modifications", []),
        "training_modifications": candidate.get("training_modifications", []),
    }

    messages = [
        {"role": "system", "content": EVOLUTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": json.dumps(clean_candidate, indent=2, ensure_ascii=False)},
    ]
    return messages


def prepare_dataset(records: list[dict], tokenizer, max_length: int = 4096) -> Dataset:
    """Tokenize evolution records into SFT dataset."""
    all_input_ids = []
    all_labels = []

    for record in records:
        messages = record_to_messages(record)
        if not messages:
            continue

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        encoded = tokenizer(
            text, truncation=True, max_length=max_length,
            return_tensors="pt", padding=False,
        )
        input_ids = encoded["input_ids"][0]
        labels = input_ids.clone()

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    print(f"  Tokenized {len(all_input_ids)}/{len(records)} records")
    return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})


class TransitionSFTCollator:
    """Pad input_ids and labels for transition SFT."""

    def __init__(self, tokenizer, max_length=4096):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_length = max_length

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = min(max(len(ids) if isinstance(ids, list) else len(ids) for ids in input_ids), self.max_length)
        padded_ids, padded_labels = [], []

        for ids, lbl in zip(input_ids, labels):
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if not isinstance(lbl, torch.Tensor):
                lbl = torch.tensor(lbl, dtype=torch.long)
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            pad_len = max_len - len(ids)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_id, dtype=ids.dtype)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)]))

        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": (torch.stack(padded_ids) != self.pad_id).long(),
        }


def train_transition(args):
    print(f"Loading evolution SFT data from {args.data_path}...")
    records = load_evolution_data(args.data_path, args.reward_threshold)

    if not records:
        print("No evolution data above threshold. Skipping.")
        return None

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing dataset...")
    dataset = prepare_dataset(records, tokenizer, max_length=args.max_length)
    print(f"  {len(dataset)} training samples")

    if len(dataset) == 0:
        print("Empty dataset. Skipping.")
        return None

    torch.cuda.empty_cache()

    # Load model (possibly with existing LoRA from inner training)
    print(f"Loading model on GPU {args.policy_gpu}: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.policy_gpu}"},
        trust_remote_code=True,
    )

    if args.resume_adapter and os.path.exists(args.resume_adapter):
        print(f"  Loading existing adapter: {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    collator = TransitionSFTCollator(tokenizer, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting Transition SFT...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Transition SFT complete!")
    return args.output_dir


def main():
    parser = argparse.ArgumentParser(description="ATS Transition SFT")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evolution_sft_data.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="Existing LoRA adapter to continue training")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reward_threshold", type=float, default=0.0,
                        help="Only train on candidates with proxy_reward >= this")
    parser.add_argument("--policy_gpu", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_transition(args)


if __name__ == "__main__":
    main()
