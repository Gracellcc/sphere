"""
ATS Reward Manager for veRL GRPO/GiGPO training.

Implements the ATS design:
  reward = training_skill.outcome_weight * outcome
         + training_skill.supervision_weight * supervision
         + training_skill.efficiency_weight * efficiency

Where:
  - outcome: 1.0 if task success, 0.0 otherwise (from environment)
  - supervision: avg of per-skill verifier scores (LLM-as-Judge on Scoring criteria)
  - efficiency: 1 - steps/max_steps (optional)

SGC gate: skip verifier when SGC > tau AND outcome=success, use SGC as proxy.

Training skill's Reward Formula is parsed at runtime, allowing the outer loop
to change reward weights dynamically.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from verl import DataProto
except ImportError:
    DataProto = None

from openai import AzureOpenAI


class ATSRewardManager:
    """ATS reward manager: verifier + training skill reward formula.

    Integrates with veRL's reward manager interface:
        __call__(data: DataProto) -> reward_tensor
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 3,
        normalize_by_length: bool = False,  # veRL interface compatibility
        # Training skill reward formula (parseable, updated by outer loop)
        outcome_weight: float = 1.0,
        supervision_weight: float = 0.3,
        efficiency_weight: float = 0.0,
        # SGC gate
        sgc_gate_tau: float = 0.6,
        # Verifier config
        verifier_model: str = "gpt-5.4",
        azure_endpoint: str = None,
        azure_api_key: str = None,
        # Skills
        skills_path: str = None,
        max_steps: int = 30,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length

        # Reward formula weights (from active training skill)
        self.outcome_weight = float(outcome_weight)
        self.supervision_weight = float(supervision_weight)
        self.efficiency_weight = float(efficiency_weight)

        # SGC gate
        self.sgc_gate_tau = float(sgc_gate_tau)

        # Max steps for efficiency calculation
        self.max_steps = max_steps

        # Load behavioral skills for scoring
        self.behavioral_skills = []
        if skills_path and os.path.exists(skills_path):
            self.behavioral_skills = self._load_behavioral_skills(skills_path)
            print(f"[ATS Reward] Loaded {len(self.behavioral_skills)} behavioral skills")

        # Verifier client (Azure OpenAI)
        self.verifier_model = verifier_model
        self.verifier_client = None
        if supervision_weight > 0:
            _endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if not _endpoint:
                raise EnvironmentError(
                    "AZURE_OPENAI_ENDPOINT must be set (via argument or environment variable)"
                )
            self.verifier_client = AzureOpenAI(
                api_key=azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
                azure_endpoint=_endpoint,
                api_version="2025-01-01-preview",
            )
            print(f"[ATS Reward] Verifier enabled: {verifier_model}")

        # Stats
        self.stats = {
            "verifier_calls": 0,
            "sgc_gate_skips": 0,
            "total_episodes": 0,
        }

    def _load_behavioral_skills(self, path: str) -> List[Dict]:
        """Load active behavioral skills from JSON file.

        Only loads skills that are type=behavioral and active (not archived).
        The outer loop may mark saturated skills as active=False.
        """
        with open(path) as f:
            data = json.load(f)

        skills = []

        def _collect(obj):
            """Recursively collect skill dicts from nested structure."""
            if isinstance(obj, list):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                # If it looks like a skill (has 'title' or 'scoring'), add it
                if "title" in obj or "scoring" in obj:
                    if obj.get("active", True):
                        skills.append(obj)
                else:
                    # Otherwise recurse into values (category dicts, etc.)
                    for v in obj.values():
                        _collect(v)

        _collect(data)

        # Validate all skills are scorable
        valid_skills = []
        for s in skills:
            ok, reason = self._validate_skill(s)
            if ok:
                valid_skills.append(s)
            else:
                print(f"[ATS Reward] WARNING: Skill '{s.get('title', '?')}' failed validation: {reason}")

        if len(valid_skills) < len(skills):
            print(f"[ATS Reward] Validated {len(valid_skills)}/{len(skills)} skills "
                  f"({len(skills) - len(valid_skills)} dropped)")

        return valid_skills

    @staticmethod
    def _validate_skill(skill: Dict) -> tuple:
        """Validate that a skill is scorable (programmatic or LLM).

        Returns (ok: bool, reason: str).
        Every skill MUST have:
          1. A 'title' field (non-empty)
          2. A 'scoring' field (non-empty) OR a 'scoring_rule' with valid type
          3. If scoring_type=='programmatic': scoring_rule must have a known type
          4. If scoring_type=='llm' or unset: scoring text must be >= 20 chars
        """
        title = skill.get("title", "").strip()
        if not title:
            return False, "missing or empty title"

        scoring_type = skill.get("scoring_type", "")
        scoring_text = skill.get("scoring", "").strip()
        scoring_rule = skill.get("scoring_rule", {})

        if scoring_type == "programmatic":
            rule_type = scoring_rule.get("type", "")
            valid_types = {"count_ratio", "pattern_before", "count_penalty",
                           "order_check", "detail_before_decision"}
            if rule_type not in valid_types:
                return False, f"programmatic skill has unknown scoring_rule.type='{rule_type}'"
            return True, ""

        # LLM or auto-detect: need meaningful scoring description
        if scoring_text and len(scoring_text) >= 20:
            return True, ""

        # Has scoring_rule but no scoring_type label → treat as programmatic
        if scoring_rule and scoring_rule.get("type"):
            return True, ""

        return False, f"LLM skill has insufficient scoring text ({len(scoring_text)} chars)"

    def update_reward_formula(self, formula: Dict[str, float]):
        """Update reward weights from active training skill.

        Called by outer loop when training skill changes.
        Formula format: {"outcome": 1.0, "supervision": 0.3, "efficiency": 0.0}
        """
        self.outcome_weight = formula.get("outcome", self.outcome_weight)
        self.supervision_weight = formula.get("supervision", self.supervision_weight)
        self.efficiency_weight = formula.get("efficiency", self.efficiency_weight)
        print(f"[ATS Reward] Updated formula: outcome={self.outcome_weight}, "
              f"supervision={self.supervision_weight}, efficiency={self.efficiency_weight}")

    # ── Scoring type inference ────────────────────────────────────────────

    @staticmethod
    def _infer_scoring_type(skill: Dict) -> str:
        """Auto-detect scoring_type from scoring text when not explicitly set.

        Looks for counting/pattern keywords that indicate programmatic scoring.
        Conservative: only returns 'programmatic' if scoring_rule is also present,
        otherwise returns 'llm' (safe default).
        """
        scoring = skill.get("scoring", "").lower()
        prog_keywords = [
            "count the number of", "count total", "count steps",
            "calls <=", "calls >=", "calls ==", "calls >",
            "total_doc_calls", "total_calls",
            "check if apis.", "check if any login",
        ]
        if any(kw in scoring for kw in prog_keywords):
            return "programmatic"
        return "llm"

    # ── Programmatic scorers ─────────────────────────────────────────────

    def _score_programmatic(self, skill: Dict, trajectory_text: str) -> float:
        """Score a skill using programmatic rules (no LLM needed).

        Dispatches based on scoring_rule.type:
          - count_ratio: count pattern occurrences normalized by apps
          - pattern_before: check if required pattern appears before target
          - count_penalty: penalize for pattern occurrences
          - order_check: check if list patterns appear before detail patterns
        """
        rule = skill.get("scoring_rule", {})
        rule_type = rule.get("type", "")

        if rule_type == "count_ratio":
            return self._score_count_ratio(rule, trajectory_text)
        elif rule_type == "pattern_before":
            return self._score_pattern_before(rule, trajectory_text)
        elif rule_type == "count_penalty":
            return self._score_count_penalty(rule, trajectory_text)
        elif rule_type == "order_check":
            return self._score_order_check(rule, trajectory_text)
        elif rule_type == "detail_before_decision":
            return self._score_detail_before_decision(rule, trajectory_text)
        else:
            return 0.5  # unknown rule type → neutral

    def _score_count_ratio(self, rule: Dict, text: str) -> float:
        """Count pattern occurrences / normalize_by. Used for doc-looping."""
        patterns = rule.get("count_patterns", [])
        total_count = sum(text.count(p) for p in patterns)
        thresholds = rule.get("thresholds", {"perfect": 1, "partial": 2})

        # Detect unique apps from trajectory
        if rule.get("normalize_by") == "unique_apps":
            app_keywords = ["spotify", "venmo", "gmail", "file_system", "admin",
                            "todoist", "simple_note", "phone", "amazon"]
            n_apps = max(1, sum(1 for app in app_keywords if app in text.lower()))
        else:
            n_apps = 1

        ratio = total_count / n_apps
        if ratio <= thresholds.get("perfect", 1):
            return 1.0
        elif ratio <= thresholds.get("partial", 2):
            return 0.5
        else:
            return 0.0

    def _score_pattern_before(self, rule: Dict, text: str) -> float:
        """Check if required patterns appear before target action."""
        required = rule.get("required_before", [])
        target = rule.get("target_action", "login")

        target_pos = text.find(target)
        if target_pos == -1:
            return 1.0  # no login needed → perfect score

        for pat in required:
            pat_pos = text.find(pat)
            if pat_pos != -1 and pat_pos < target_pos:
                return rule.get("score_if_present", 1.0)

        return rule.get("score_if_absent", 0.0)

    def _score_count_penalty(self, rule: Dict, text: str) -> float:
        """Penalize for pattern occurrences (e.g., recovery detours)."""
        patterns = rule.get("penalty_patterns", [])
        total = sum(text.lower().count(p) for p in patterns)
        thresholds = rule.get("thresholds", {"zero": 0, "one": 1})

        if total <= thresholds.get("zero", 0):
            return 1.0
        elif total <= thresholds.get("one", 1):
            return 0.5
        else:
            return 0.0

    def _score_order_check(self, rule: Dict, text: str) -> float:
        """Check if list/index patterns appear before detail patterns."""
        list_pats = rule.get("list_patterns", [])
        detail_pats = rule.get("detail_patterns", [])

        # Find earliest list call and earliest detail call
        first_list = len(text)
        for pat in list_pats:
            pos = text.find(pat)
            if pos != -1:
                first_list = min(first_list, pos)

        first_detail = len(text)
        for pat in detail_pats:
            pos = text.find(pat)
            if pos != -1:
                first_detail = min(first_detail, pos)

        if first_detail == len(text):
            return 1.0  # no detail calls → OK
        if first_list == len(text):
            return 0.0  # detail called but no list call
        return 1.0 if first_list < first_detail else 0.0

    def _score_detail_before_decision(self, rule: Dict, text: str) -> float:
        """Check if detail endpoints were called (vs only list endpoints)."""
        detail_pats = rule.get("detail_patterns", [])
        found = sum(1 for p in detail_pats if p in text)
        return min(1.0, found / max(1, len(detail_pats) / 2))

    # ── LLM Verifier ──────────────────────────────────────────────────

    VERIFIER_SYSTEM_PROMPT = (
        "You are a strict trajectory scoring judge. Your ONLY job is to output a score and reason.\n\n"
        "RULES:\n"
        "1. You MUST output exactly two lines, nothing else.\n"
        "2. Line 1 MUST be: SCORE: <decimal between 0.0 and 1.0>\n"
        "3. Line 2 MUST be: REASON: <one sentence explanation>\n"
        "4. If the trajectory is too short or unclear to judge, output SCORE: 0.5\n"
        "5. If the skill criteria don't apply to this trajectory, output SCORE: 0.5\n"
        "6. NEVER output an empty response. NEVER skip the SCORE line.\n\n"
        "Example output:\n"
        "SCORE: 0.75\n"
        "REASON: The agent followed most criteria but missed pagination on the second endpoint."
    )

    def _call_verifier(self, skill: Dict, trajectory_text: str, max_retries: int = 2) -> float:
        """Call LLM verifier to score trajectory against skill's Scoring criteria.

        Returns 0-1 score. Uses system prompt for format enforcement,
        retries with nudge on parse failure, falls back to heuristic scoring.
        """
        scoring = skill.get("scoring", skill.get("principle", ""))
        title = skill.get("title", "unknown")

        user_msg = (
            f"## Behavioral Skill: {title}\n\n"
            f"## Scoring Criteria (use these to determine the score):\n{scoring}\n\n"
            f"## Agent Trajectory (truncated to key steps):\n{trajectory_text[:4000]}\n\n"
            f"---\nBased on the scoring criteria above, evaluate this trajectory.\n"
            f"Output EXACTLY two lines:\nSCORE: <0.0 to 1.0>\nREASON: <one sentence>"
        )

        for attempt in range(max_retries):
            try:
                response = self.verifier_client.chat.completions.create(
                    model=self.verifier_model,
                    messages=[
                        {"role": "system", "content": self.VERIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=150,
                    temperature=0.0,
                )
                text = (response.choices[0].message.content or "").strip()

                # Guard: empty response
                if not text:
                    if attempt < max_retries - 1:
                        user_msg += "\n\n[REMINDER: You must output SCORE: and REASON: lines.]"
                        continue
                    return self._fallback_score(skill, trajectory_text)

                # Parse SCORE: pattern first, then any float
                match = re.search(r"SCORE:\s*([\d.]+)", text, re.IGNORECASE)
                if match is None:
                    match = re.search(r"(\d+\.\d+)", text)
                if match is None:
                    if attempt < max_retries - 1:
                        user_msg += "\n\n[REMINDER: You must output SCORE: and REASON: lines.]"
                        continue
                    return self._fallback_score(skill, trajectory_text)

                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"[ATS Reward] Verifier failed for {title}: {e}")
                    return self._fallback_score(skill, trajectory_text)

        return self._fallback_score(skill, trajectory_text)

    def _fallback_score(self, skill: Dict, trajectory_text: str) -> float:
        """Heuristic fallback when LLM verifier fails.

        Extracts verifiable signals from the scoring description and checks
        against the trajectory. Returns a coarse score clamped to [0.2, 0.8].
        """
        scoring = skill.get("scoring", "").lower()
        text_lower = trajectory_text.lower()
        signals = []

        # Signal 1: API patterns mentioned in scoring → check if present in trajectory
        api_patterns = re.findall(
            r"(show_\w+|get_\w+|list_\w+|search_\w+|create_\w+|update_\w+|"
            r"like_\w+|login|complete_task|reset_password)",
            scoring,
        )
        if api_patterns:
            found = sum(1 for p in api_patterns if p in text_lower)
            signals.append(min(1.0, found / max(1, len(api_patterns) * 0.5)))

        # Signal 2: "check if X before Y" → simple order check
        before_match = re.search(
            r"check if (\w[\w_.]+).*?(?:before|prior to|preceded).*?(\w[\w_.]+)",
            scoring,
        )
        if before_match:
            a, b = before_match.group(1), before_match.group(2)
            pos_a, pos_b = text_lower.find(a), text_lower.find(b)
            if pos_b == -1:
                signals.append(1.0)  # target not present
            elif pos_a != -1 and pos_a < pos_b:
                signals.append(1.0)
            else:
                signals.append(0.0)

        # Signal 3: Did agent call complete_task?
        signals.append(0.7 if "complete_task" in text_lower else 0.3)

        if signals:
            raw = sum(signals) / len(signals)
            return max(0.2, min(0.8, raw))
        return 0.5

    def _compute_supervision(
        self,
        trajectory_text: str,
        outcome_success: bool,
        sgc_score: float = 0.0,
        skills_used: Optional[List[str]] = None,
    ) -> float:
        """Compute supervision reward from behavioral skills.

        Only scores skills that were actually injected during the episode.
        Uses SGC gate: skip verifier if SGC > tau AND success.
        """
        if not self.behavioral_skills:
            return 0.0

        # SGC gate
        if outcome_success and sgc_score > self.sgc_gate_tau:
            self.stats["sgc_gate_skips"] += 1
            return sgc_score  # Use SGC as proxy

        # Filter to only skills that were injected in this episode
        if skills_used:
            used_set = set(skills_used)
            skills_to_score = [
                s for s in self.behavioral_skills
                if s.get("title", "") in used_set
            ]
            if not skills_to_score:
                # Fallback: if no match by title, score all (shouldn't happen)
                skills_to_score = self.behavioral_skills
        else:
            # No skills_used info (e.g. no Sphere) → score all
            skills_to_score = self.behavioral_skills

        # Split into programmatic vs LLM skills (auto-detect if not labeled)
        prog_skills = []
        llm_skills = []
        for s in skills_to_score:
            stype = s.get("scoring_type") or self._infer_scoring_type(s)
            if stype == "programmatic" and s.get("scoring_rule"):
                prog_skills.append(s)
            else:
                llm_skills.append(s)

        scores = []

        # Programmatic: instant, no API calls
        for skill in prog_skills:
            score = self._score_programmatic(skill, trajectory_text)
            scores.append(score)
            self.stats["programmatic_scores"] = self.stats.get("programmatic_scores", 0) + 1

        # LLM: parallel verifier calls
        if llm_skills:
            with ThreadPoolExecutor(max_workers=min(8, len(llm_skills))) as pool:
                futures = {
                    pool.submit(self._call_verifier, skill, trajectory_text): skill
                    for skill in llm_skills
                }
                for future in as_completed(futures):
                    scores.append(future.result())
                    self.stats["verifier_calls"] += 1

        return float(np.mean(scores)) if scores else 0.0

    def compute_episode_reward(
        self,
        outcome_success: bool,
        n_steps: int,
        trajectory_text: str = "",
        sgc_score: float = 0.0,
        tgc_score: float = 0.0,
        skills_used: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute full ATS reward for one episode.

        Returns dict with component breakdown.
        """
        self.stats["total_episodes"] += 1

        # Outcome
        outcome = 1.0 if outcome_success else 0.0

        # Supervision (verifier or SGC proxy) — only score injected skills
        supervision = 0.0
        if self.supervision_weight > 0 and trajectory_text:
            supervision = self._compute_supervision(
                trajectory_text, outcome_success, sgc_score, skills_used
            )

        # Efficiency
        efficiency = 1.0 - (n_steps / self.max_steps) if self.max_steps > 0 else 0.0

        # Combined reward using training skill's formula
        reward = (
            self.outcome_weight * outcome
            + self.supervision_weight * supervision
            + self.efficiency_weight * efficiency
        )

        return {
            "reward": float(reward),
            "outcome": float(outcome),
            "supervision": float(supervision),
            "efficiency": float(efficiency),
            "sgc_score": float(sgc_score),
            "tgc_score": float(tgc_score),
        }

    def __call__(self, data: DataProto, return_dict=False):
        """veRL reward manager interface.

        Follows SelfSkill's EpisodeRewardManager pattern:
          - Returns pre-computed rm_scores if present (veRL pipeline compat)
          - Reads episode_rewards (accumulated env rewards) from non_tensor_batch
          - Reads episode_lengths from non_tensor_batch
          - Reads trajectory_text from non_tensor_batch (full multi-turn history)
          - Reads sgc_score from non_tensor_batch (SGC gate for verifier skip)
          - Places final reward at last valid token position

        Data flow (set by veRL's gather_rollout_data):
          data.non_tensor_batch['episode_rewards'] = float (sum of env step rewards)
          data.non_tensor_batch['episode_lengths'] = int (number of active steps)
          data.non_tensor_batch['trajectory_text'] = str (full multi-turn trajectory)
          data.non_tensor_batch['sgc_score'] = float (SGC confidence, 0-1)
          data.non_tensor_batch['skills_used'] = list[str] (skill names injected this episode)
        """
        # Passthrough pre-computed scores (veRL pipeline compatibility)
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        prompt_ids = data.batch["prompts"]
        device = prompt_ids.device

        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )

        all_rewards = []
        already_printed = 0
        print(f"[ATS Reward] computing rewards for {len(data)} items...", flush=True)

        # Cache: same trajectory_text → same supervision score (avoid redundant API calls)
        # veRL expands data per-step, so 32 trajectories × ~18 steps = 576 items,
        # but supervision only needs to be computed once per unique trajectory.
        _supervision_cache = {}

        for i in range(len(data)):
            data_item = data[i]

            # ── Extract episode info (veRL rollout loop format) ──────────
            if "episode_rewards" not in data_item.non_tensor_batch:
                print(f"[ATS Reward] WARNING: episode_rewards missing for item {i}, defaulting to 0.0")
            episode_reward = float(
                data_item.non_tensor_batch.get("episode_rewards", 0.0)
            )
            episode_length = int(
                data_item.non_tensor_batch.get("episode_lengths", 1)
            )
            outcome_success = episode_reward > 0.5

            # ── Get trajectory text ──
            trajectory_text = ""
            if self.supervision_weight > 0:
                trajectory_text = str(
                    data_item.non_tensor_batch.get("trajectory_text", "")
                )
                if not trajectory_text:
                    try:
                        response_ids = data_item.batch["responses"]
                        prompt_length = data_item.batch["prompts"].shape[-1]
                        attn = data_item.batch["attention_mask"][prompt_length:]
                        valid_ids = response_ids[attn.bool()]
                        trajectory_text = self.tokenizer.decode(
                            valid_ids, skip_special_tokens=True
                        )
                    except Exception:
                        trajectory_text = ""

            # ── Get SGC score and TGC score ─
            sgc_score = float(
                data_item.non_tensor_batch.get("sgc_score", 0.0)
            )
            tgc_score = float(
                data_item.non_tensor_batch.get("tgc_score", 0.0)
            )

            # ── Get skills_used ─
            skills_used = data_item.non_tensor_batch.get("skills_used", None)
            if isinstance(skills_used, np.ndarray):
                skills_used = skills_used.tolist()

            # ── Compute ATS reward (with supervision cache) ──────────────
            # Cache key: hash of trajectory text (same trajectory → same supervision)
            traj_key = hash(trajectory_text) if trajectory_text else None

            if traj_key is not None and traj_key in _supervision_cache:
                # Reuse cached supervision, recompute only outcome/efficiency
                cached_sup = _supervision_cache[traj_key]
                reward_info = self.compute_episode_reward(
                    outcome_success=outcome_success,
                    n_steps=episode_length,
                    trajectory_text="",  # empty → skip verifier
                    sgc_score=sgc_score,
                    tgc_score=tgc_score,
                    skills_used=skills_used,
                )
                # Override supervision with cached value
                reward_info["supervision"] = cached_sup
                reward_info["reward"] = (
                    self.outcome_weight * reward_info["outcome"]
                    + self.supervision_weight * cached_sup
                    + self.efficiency_weight * reward_info["efficiency"]
                )
            else:
                reward_info = self.compute_episode_reward(
                    outcome_success=outcome_success,
                    n_steps=episode_length,
                    trajectory_text=trajectory_text,
                    sgc_score=sgc_score,
                    tgc_score=tgc_score,
                    skills_used=skills_used,
                )
                if traj_key is not None:
                    _supervision_cache[traj_key] = reward_info["supervision"]

            reward = reward_info["reward"]
            all_rewards.append(reward_info)

            # ── Place reward at last valid token (veRL convention) ───────
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = int(
                data_item.batch["attention_mask"][prompt_length:].sum()
            )
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = torch.tensor(
                    reward, dtype=torch.float32, device=device,
                )

            # Debug printing
            if already_printed < self.num_examine:
                already_printed += 1
                print(
                    f"[ATS Reward] episode {i}: "
                    f"outcome={reward_info['outcome']:.0f}, "
                    f"supervision={reward_info['supervision']:.2f}, "
                    f"efficiency={reward_info['efficiency']:.2f}, "
                    f"sgc={sgc_score:.2f}, tgc={tgc_score:.2f}, "
                    f"total={reward:.3f}"
                )

        n_cached = len(data) - len(_supervision_cache)
        print(f"[ATS Reward] done: {len(_supervision_cache)} unique trajectories, {n_cached} cache hits", flush=True)

        if return_dict:
            # reward_extra_info values MUST be batch-sized arrays (not dicts/scalars)
            # because veRL does: non_tensor_batch.update({k: np.array(v) ...})
            # and then indexes val[i] — 0-d arrays from np.array(dict) would crash.
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "reward_outcome": np.array([r["outcome"] for r in all_rewards], dtype=np.float32),
                    "reward_supervision": np.array([r["supervision"] for r in all_rewards], dtype=np.float32),
                    "reward_efficiency": np.array([r["efficiency"] for r in all_rewards], dtype=np.float32),
                    "reward_total": np.array([r["reward"] for r in all_rewards], dtype=np.float32),
                },
            }
        return reward_tensor

    def get_stats(self) -> Dict[str, Any]:
        """Return reward manager statistics."""
        return {
            **self.stats,
            "verifier_call_rate": (
                self.stats["verifier_calls"]
                / max(1, self.stats["total_episodes"])
            ),
            "sgc_skip_rate": (
                self.stats["sgc_gate_skips"]
                / max(1, self.stats["total_episodes"])
            ),
        }
