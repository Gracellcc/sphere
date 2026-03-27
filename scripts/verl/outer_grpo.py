"""
Outer GRPO: Train model's skill evolution capability (Phase 2).

Design doc: "GRPO更新θ的skill修改能力"
  J(θ) = J_inner(θ) + λ_outer · J_outer(θ)

Inner GRPO teaches the model to solve tasks.
Outer GRPO teaches the model to generate good skill modifications.

Flow:
  1. Build evolution prompt from diagnostics
  2. Generate G candidates with per-token log probs
  3. Proxy eval each candidate → reward
  4. GRPO advantage = reward_i - mean(rewards)
  5. Clipped surrogate policy gradient update
  6. Best candidate → update skill bank

Usage:
  Called by ats_pipeline.py during Phase 2 outer loop.
  Runs sequentially with proxy eval (not concurrent).
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class OuterGRPO:
    """GRPO for skill evolution capability.

    Loads the current model, generates skill modification candidates,
    and does a policy gradient step using proxy eval rewards.
    """

    def __init__(
        self,
        model_path: str,
        lambda_outer: float = 0.1,
        clip_eps: float = 0.2,
        lr: float = 1e-6,
        max_new_tokens: int = 2048,
        gpu_id: int = 0,
    ):
        self.device = f"cuda:{gpu_id}"
        self.lambda_outer = lambda_outer
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens

        print(f"[OuterGRPO] Loading model: {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        print(f"[OuterGRPO] Model loaded. lambda_outer={lambda_outer}, lr={lr}")

    @torch.no_grad()
    def generate_candidate(
        self, messages: List[Dict], temperature: float = 0.7
    ) -> Dict:
        """Generate one candidate and return text + log probs + token ids.

        Args:
            messages: Chat messages [system, user] for evolution prompt.
            temperature: Sampling temperature.

        Returns:
            Dict with keys: text, generated_ids, old_log_probs, input_ids
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        gen_ids = outputs.sequences[0, input_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Per-token log probs from generation scores
        log_probs = []
        for t, score in enumerate(outputs.scores):
            if t >= len(gen_ids):
                break
            lp = F.log_softmax(score[0].float(), dim=-1)
            log_probs.append(lp[gen_ids[t]].item())

        return {
            "text": gen_text,
            "generated_ids": gen_ids.cpu(),
            "old_log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "input_ids": inputs.input_ids[0].cpu(),
        }

    def generate_candidates(
        self, messages: List[Dict], G: int = 3, base_temperature: float = 0.7
    ) -> List[Dict]:
        """Generate G candidates with increasing temperature.

        Args:
            messages: Chat messages for evolution prompt.
            G: Number of candidates.
            base_temperature: Starting temperature (increases by 0.1 per candidate).

        Returns:
            List of candidate dicts with text, generated_ids, old_log_probs, input_ids.
        """
        candidates = []
        for g in range(G):
            temp = base_temperature + g * 0.1
            print(f"[OuterGRPO] Generating candidate {g+1}/{G} (temp={temp:.1f})")
            try:
                cand = self.generate_candidate(messages, temperature=temp)
                cand["candidate_id"] = g
                cand["source"] = "model_grpo"
                candidates.append(cand)
                print(f"  Generated {len(cand['generated_ids'])} tokens")
            except Exception as e:
                print(f"  Generation failed: {e}")
        return candidates

    def _compute_token_log_probs(
        self, input_ids: torch.Tensor, generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute log probs of generated tokens (with gradients).

        Args:
            input_ids: Prompt token ids [seq_len].
            generated_ids: Generated token ids [gen_len].

        Returns:
            Per-token log probs [gen_len] with gradient attached.
        """
        full_ids = torch.cat([input_ids, generated_ids], dim=0).unsqueeze(0)
        full_ids = full_ids.to(self.device)

        outputs = self.model(full_ids)

        # Logits at positions [input_len-1 .. total_len-2] predict tokens [input_len .. total_len-1]
        input_len = input_ids.shape[0]
        logits = outputs.logits[0, input_len - 1 : -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)

        gen_ids_device = generated_ids.to(self.device)
        token_log_probs = log_probs.gather(
            1, gen_ids_device.unsqueeze(1)
        ).squeeze(1)

        return token_log_probs

    def train_step(self, candidates: List[Dict]) -> Tuple[float, float, int]:
        """One GRPO update step on the generated candidates.

        Each candidate must have: input_ids, generated_ids, old_log_probs, reward.

        Args:
            candidates: List of candidate dicts with rewards filled in.

        Returns:
            (loss, mean_reward, best_candidate_index)
        """
        rewards = torch.tensor([c["reward"] for c in candidates], dtype=torch.float32)

        # GRPO: group-relative advantage (no std normalization, like SAGE)
        advantages = rewards - rewards.mean()

        # Skip if all rewards identical
        if rewards.std() < 1e-8:
            print("[OuterGRPO] All rewards identical, skipping update")
            best_idx = 0
            return 0.0, rewards.mean().item(), best_idx

        print(f"[OuterGRPO] Rewards: {rewards.tolist()}")
        print(f"[OuterGRPO] Advantages: {advantages.tolist()}")

        self.optimizer.zero_grad()
        total_loss = 0.0

        for i, (c, adv) in enumerate(zip(candidates, advantages)):
            input_ids = c["input_ids"]
            gen_ids = c["generated_ids"]
            old_lp = c["old_log_probs"].to(self.device)

            # Current policy log probs (with gradients)
            cur_lp = self._compute_token_log_probs(input_ids, gen_ids)

            # Truncate to same length (in case of minor mismatch)
            min_len = min(len(cur_lp), len(old_lp))
            cur_lp = cur_lp[:min_len]
            old_lp = old_lp[:min_len]

            # Importance sampling ratio
            ratio = torch.exp(cur_lp - old_lp)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

            # Clipped surrogate (maximize advantage, so negate for loss)
            adv_device = adv.to(self.device)
            surr1 = ratio * adv_device
            surr2 = clipped_ratio * adv_device
            policy_loss = -torch.min(surr1, surr2).mean()

            # Scale by lambda_outer and accumulate gradient
            scaled_loss = self.lambda_outer * policy_loss / len(candidates)
            scaled_loss.backward()
            total_loss += scaled_loss.item()

        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        best_idx = rewards.argmax().item()
        print(
            f"[OuterGRPO] Loss: {total_loss:.6f}, "
            f"Mean reward: {rewards.mean():.4f}, "
            f"Best candidate: {best_idx} (reward={rewards[best_idx]:.4f})"
        )

        return total_loss, rewards.mean().item(), best_idx

    def save(self, output_path: str):
        """Save updated model to HF format."""
        os.makedirs(output_path, exist_ok=True)
        print(f"[OuterGRPO] Saving model to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()
        print("[OuterGRPO] GPU memory freed")


def build_evolution_messages(prompt: str) -> List[Dict]:
    """Wrap evolution prompt into chat messages format."""
    return [
        {
            "role": "system",
            "content": (
                "You are a skill evolution expert for an AI agent training system. "
                "Analyze diagnostics and propose skill modifications as valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]
