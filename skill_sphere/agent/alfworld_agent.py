"""ALFWorld agent with Skill Sphere retrieval.

Implements the full Skill Sphere inference loop as described in the framework:
1. Maintain a policy intent point `t` on the sphere that evolves per step
2. Confidence-guided dynamic injection (only inject when uncertain)
3. Per-step retrieval based on current state, not just initial task
4. Slerp combination + complementarity selection
5. General skills always available + sphere selects task-specific skills

Supports three modes:
- "sphere": Full Skill Sphere dynamic retrieval (ours)
- "skillrl": SkillRL-style template retrieval (baseline)
- "none": No skill injection (baseline)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from skill_sphere.agent.llm_client import LLMClient
from skill_sphere.skill_bank.skill_sphere import SkillSphere
from skill_sphere.geometry.sphere import slerp
from skill_sphere.injection.dynamic_inject import DynamicInjector
from skill_sphere.injection.confidence import compute_logit_confidence, SphereConfidence
from skill_sphere.injection.intent_tracker import IntentTracker


# --- Task type detection ---

def detect_task_type(task_description: str) -> str:
    """Detect ALFWorld task type from task description."""
    goal = task_description.lower()

    if ("examine" in goal or "look at" in goal) and ("lamp" in goal or "light" in goal):
        return "look_at_obj_in_light"
    if "clean" in goal:
        return "clean"
    if "hot" in goal or "heat" in goal:
        return "heat"
    if "cool" in goal or "cold" in goal:
        return "cool"
    if "two" in goal:
        return "pick_and_place"
    return "pick_and_place"


# --- Action parsing ---

def parse_action(raw_output: str) -> tuple[str, str, bool]:
    """Parse agent output to extract think and action."""
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    action_match = re.search(r"<action>(.*?)</action>", raw_output, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().lower()
        return thinking, action, True
    else:
        lines = raw_output.strip().split("\n")
        action = lines[-1].strip().lower() if lines else ""
        return thinking, action, False


def match_to_admissible(action: str, admissible: list[str]) -> str:
    """Match a parsed action to the closest admissible action."""
    if not admissible:
        return action

    action_lower = action.lower().strip()
    for a in admissible:
        if a.lower().strip() == action_lower:
            return a

    for a in admissible:
        if action_lower in a.lower() or a.lower() in action_lower:
            return a

    action_words = set(action_lower.split())
    best_score = -1
    best_action = admissible[0]
    for a in admissible:
        a_words = set(a.lower().split())
        score = len(action_words & a_words)
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


# --- Confidence estimation ---

def estimate_confidence(raw_output: str, admissible: list[str]) -> float:
    """Estimate agent's confidence from its output.

    Heuristic proxy for true logit-based confidence:
    - Exact match to admissible → high confidence
    - Has <think> and <action> tags → moderate confidence
    - Hedging language ("maybe", "not sure", "try") → lower confidence
    - Repeating previous failed action → low confidence
    """
    confidence = 0.5  # base

    # Has proper tags?
    has_think = "<think>" in raw_output and "</think>" in raw_output
    has_action = "<action>" in raw_output and "</action>" in raw_output
    if has_think and has_action:
        confidence += 0.2

    # Action matches admissible exactly?
    action_match = re.search(r"<action>(.*?)</action>", raw_output, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().lower()
        if any(a.lower().strip() == action for a in admissible):
            confidence += 0.15

    # Hedging language?
    think_text = ""
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).lower()

    hedging_words = ["maybe", "not sure", "uncertain", "try", "perhaps",
                     "might", "could be", "don't know", "unclear"]
    for word in hedging_words:
        if word in think_text:
            confidence -= 0.1
            break

    return max(0.0, min(1.0, confidence))


# --- Prompt templates ---

SYSTEM_PROMPT_BASE = """You are an expert agent operating in the ALFRED Embodied Environment. \
You must complete household tasks by taking actions step by step.

You should first reason step-by-step about the current situation. \
This reasoning MUST be enclosed within <think> </think> tags.
Then choose an admissible action and present it within <action> </action> tags.

Example output format:
<think>
I need to find the apple. Let me check the countertop first.
</think>
<action>
go to countertop 1
</action>"""

SKILL_INJECTION_TEMPLATE = """
## Skill Guidance{confidence_note}

{skill_text}"""

STEP_PROMPT_TEMPLATE = """Task: {task_description}

{history_block}
Current step: {step_num}
Current observation: {observation}

Admissible actions: [{admissible_actions}]"""

# SkillRL-native single-message template (matches their GRPO training format exactly)
SKILLRL_NATIVE_TEMPLATE_NO_HIS = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SKILLRL_NATIVE_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SKILLRL_NATIVE_TEMPLATE_WITH_MEMORY = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


# --- Agent ---

@dataclass
class StepLog:
    """Log for a single agent step."""
    step: int
    observation: str
    thinking: str
    action: str
    action_valid: bool
    reward: float
    confidence: float = 0.0
    skills_injected: bool = False
    skills_used: list[str] = field(default_factory=list)
    skill_scores: list[float] = field(default_factory=list)
    injection_strength: float = 0.0
    # Sphere geometric signals
    drift_rate: float = 0.0
    drift_norm: float = 0.0
    adaptive_momentum: float = 0.0
    isolation_score: float = 0.0
    in_uncharted: bool = False
    gamma: float = 1.0
    regime: str = "neutral"
    alignment: float = 1.0
    # Full prompt/response for FAC feature extraction
    system_prompt: str = ""
    user_prompt: str = ""
    raw_output: str = ""


class ALFWorldAgent:
    """Agent that uses Skill Sphere for ALFWorld tasks.

    Implements the full sphere inference loop:
    - Policy intent point `t` on sphere, evolving per step
    - Confidence-guided dynamic injection
    - Per-step retrieval when uncertain
    - General skills always available as base
    """

    def __init__(
        self,
        llm: LLMClient,
        skill_sphere: SkillSphere | None = None,
        mode: str = "sphere",
        max_history: int = 5,
        intent_momentum: float = 0.3,
        sigma: float = 1.5,
        min_inject_strength: float = 0.15,
        confidence_temperature: float = 4.0,
        enable_bridge: bool = False,
        bridge_k: int = 1,
    ):
        """
        Args:
            llm: LLM client for generating actions.
            skill_sphere: SkillSphere instance.
            mode: "sphere" (our method), "skillrl" (template baseline), "none" (no skills).
            max_history: Maximum number of recent steps to include in prompt.
            intent_momentum: How fast the intent point t updates (0=static, 1=instant).
            sigma: Width of distance decay Gaussian for injection weight (in d_typical units).
                   σ=1.5 means skills at 1.5× the typical inter-skill distance still get
                   moderate weight (exp(-1) ≈ 0.37).
            min_inject_strength: Floor for injection base_strength. Even at maximum
                   model confidence, base_strength >= this value. Prevents confidence
                   from completely suppressing skill injection on untrained models.
            confidence_temperature: Temperature for logit-based confidence estimation.
                   Higher values flatten the distribution, reducing overconfidence.
        """
        self.llm = llm
        self.skill_sphere = skill_sphere
        self.mode = mode
        self.max_history = max_history
        self.intent_momentum = intent_momentum
        self.min_inject_strength = min_inject_strength
        self.confidence_temperature = confidence_temperature

        # MD-style dynamic injector — no hard threshold, continuous formula only
        self.injector = DynamicInjector(
            sigma=sigma,
            min_weight=0.05,
            min_inject_strength=min_inject_strength,
            max_skills=5,
            relevance_k=10,
            redundancy_threshold=0.85,
            enable_bridge=enable_bridge,
            bridge_k=bridge_k,
        )

        # Strategy intent point tracker (MD 6.3)
        self.intent_tracker = IntentTracker()
        # Sphere-Geometric Confidence (replaces logit entropy)
        self.sgc = SphereConfidence()

    def run_episode(
        self,
        task_description: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int = 50,
    ) -> tuple[bool, list[StepLog]]:
        """Run a complete episode with dynamic sphere-guided skill injection.

        The sphere mode maintains a policy intent point `t` that evolves:
        - t starts as the encoded task description
        - Each step, t is updated based on the new observation
        - When confidence < threshold, skills are retrieved near t
        - General skills provide base guidance throughout
        """
        task_type = detect_task_type(task_description)

        # --- Mode dispatch ---
        if self.mode == "skillrl":
            return self._run_episode_skillrl(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
            )
        elif self.mode == "skillrl_fair":
            return self._run_episode_skillrl_fair(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
            )
        elif self.mode == "embed_topk":
            return self._run_episode_embed_topk(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
            )
        elif self.mode == "skillrl_native":
            return self._run_episode_skillrl_native(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
            )
        elif self.mode == "skillrl_native_sphere":
            return self._run_episode_skillrl_native(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
                use_sphere=True,
            )
        elif self.mode == "none":
            return self._run_episode_static(
                task_description, "", initial_observation,
                admissible_commands, step_fn, max_steps,
            )
        else:  # sphere
            return self._run_episode_sphere(
                task_description, task_type, initial_observation,
                admissible_commands, step_fn, max_steps,
            )

    def _run_episode_sphere(
        self,
        task_description: str,
        task_type: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """Full Skill Sphere inference loop — Unified Sphere Pipeline.

        Sphere-unique features:
        1. Fixed intent momentum (0.3) with drift logging via IntentTracker
        2. Boost-only γ factor (≥1.0) when agent is stuck (MD 6.2 + 6.3)
        3. Coverage-aware uncharted territory detection (MD 4.5)
        4. Skills re-ranked by distance to Fréchet mean (MD 7.①)
        5. Skill rotation: penalize recently-injected skills for diversity
        """
        if self.skill_sphere is None or self.skill_sphere.encoder is None:
            return self._run_episode_static(
                task_description, "", initial_observation,
                admissible_commands, step_fn, max_steps,
            )

        import torch

        # Encode task description → initialize policy intent point t
        t = self.skill_sphere.encoder.encode_query(task_description)

        # Calibrate sphere-adaptive components (once per episode if needed)
        cal_stats = self.injector.calibrate(self.skill_sphere.vectors)
        self.intent_tracker.calibrate(self.injector.retriever.d_typical)

        # Set bridge skill indices if bridge selection is enabled
        if self.injector.enable_bridge:
            bridge_indices = [
                i for i, s in enumerate(self.skill_sphere.skills)
                if s.metadata.get("skill_id", "").startswith("bridge_")
            ]
            if bridge_indices:
                self.injector.set_bridge_indices(bridge_indices)
        self.intent_tracker.reset(t)
        self.sgc.reset()

        # --- Environment-aware filtering (unified v2) ---
        # Keep only ALFWorld-relevant categories; remove AppWorld/ScienceWorld skills
        ALFWORLD_CATEGORIES = {
            "general", "common_mistakes",
            "clean", "cool", "heat", "pick", "pick_two", "examine",
            "pick_and_place", "look_at_obj_in_light",
        }
        filtered_indices = self._get_env_filtered_indices(ALFWORLD_CATEGORIES)
        if filtered_indices is not None and len(filtered_indices) < len(self.skill_sphere.skills):
            filtered_vectors = torch.stack([self.skill_sphere.vectors[i] for i in filtered_indices])
            print(f"  [unified v2] Filtered {len(self.skill_sphere.skills)}→{len(filtered_indices)} skills for ALFWorld")
        else:
            filtered_vectors = None
            filtered_indices = None

        # Get general skills (always available as base context)
        general_text = self._format_general_skills(task_type, task_description)

        history: list[tuple[str, str]] = []
        logs: list[StepLog] = []
        observation = initial_observation
        success = False
        prev_confidence = 0.5
        prev_actions: list[str] = []
        prev_drift: float = 0.0
        prev_coherence: float = 1.0
        prev_stability: float = 1.0
        recently_used_skills: list[int] = []  # For skill rotation

        for step_num in range(1, max_steps + 1):
            system_prompt = SYSTEM_PROMPT_BASE
            if general_text:
                system_prompt += SKILL_INJECTION_TEMPLATE.format(
                    skill_text=general_text,
                    confidence_note=" (General Principles)",
                )

            history_block = self._format_history(history)
            admissible_str = ", ".join(f"'{a}'" for a in admissible_commands)
            user_prompt = STEP_PROMPT_TEMPLATE.format(
                task_description=task_description,
                history_block=history_block,
                step_num=step_num,
                observation=observation,
                admissible_actions=admissible_str,
            )

            # --- Sphere Injection ---
            in_loop = (
                len(prev_actions) >= 2
                and len(set(prev_actions[-2:])) == 1
            )
            force = in_loop or step_num == 1

            # Sphere-Geometric Confidence (SGC) replaces logit entropy
            isolation = self.injector.compute_isolation(t, self.skill_sphere.vectors)
            sgc_signals = self.sgc.compute(
                coherence=prev_coherence,
                stability=prev_stability,
                isolation_score=isolation,
            )
            effective_confidence = 0.1 if in_loop else sgc_signals.sgc

            # Skill rotation: when looping or low confidence, penalize recent skills
            rotation_active = in_loop or effective_confidence < 0.4
            rotation_list = recently_used_skills if rotation_active else None

            skill_text, injected, strength, inj_result = self._inject_with_md_formula(
                t, effective_confidence, force=force, drift_rate=prev_drift,
                recently_used=rotation_list,
                filtered_vectors=filtered_vectors,
                filtered_indices=filtered_indices,
            )

            # Uncharted territory fallback (MD 4.5)
            if inj_result and inj_result.in_uncharted and not inj_result.should_inject:
                skill_text = general_text if general_text else ""

            if skill_text:
                if in_loop:
                    skill_text += "\n\nIMPORTANT: You are repeating the same action. Try a DIFFERENT action from the admissible list."
                conf_note = f" (confidence={effective_confidence:.2f}, w={strength:.2f})"
                user_prompt += SKILL_INJECTION_TEMPLATE.format(
                    skill_text=skill_text,
                    confidence_note=conf_note,
                )

            raw_output = self.llm.generate_action(system_prompt, user_prompt)
            thinking, action, action_valid = parse_action(raw_output)
            confidence = effective_confidence  # Use SGC (no logprobs needed)
            action = match_to_admissible(action, admissible_commands)
            prev_actions.append(action.lower().strip())

            obs_new, reward, done, admissible_new = step_fn(action)

            # Update intent point with adaptive momentum (sphere-calibrated)
            context_text = f"{task_description}. Current: {obs_new[:200]}"
            try:
                context_vec = self.skill_sphere.encoder.encode_query(context_text)
                drift_info = self.intent_tracker.update(context_vec)
                t = slerp(
                    t.unsqueeze(0),
                    context_vec.unsqueeze(0),
                    drift_info.alpha,  # Adaptive: high drift → fast chase, low drift → stable
                ).squeeze(0)
                prev_drift = drift_info.drift_rate
                prev_coherence = drift_info.coherence
                prev_stability = drift_info.stability
            except Exception:
                prev_drift = 0.0

            # Track recently injected skills for rotation
            if inj_result and inj_result.selected_indices:
                recently_used_skills.extend(inj_result.selected_indices)
                recently_used_skills = recently_used_skills[-15:]  # Keep last ~5 steps

            retrieved_skill_names = []
            retrieved_skill_scores = []
            if inj_result and inj_result.selected_indices:
                for idx, score in zip(inj_result.selected_indices, inj_result.injection_weights):
                    retrieved_skill_names.append(self.skill_sphere.skills[idx].name)
                    retrieved_skill_scores.append(score)

            # Extract sphere geometric signals
            _drift_rate = drift_info.drift_rate if 'drift_info' in dir() and drift_info else prev_drift
            _drift_norm = drift_info.drift_norm if 'drift_info' in dir() and drift_info else 0.0
            _adaptive_momentum = drift_info.alpha if 'drift_info' in dir() and drift_info else 0.0
            _isolation = inj_result.isolation_score if inj_result else 0.0
            _uncharted = inj_result.in_uncharted if inj_result else False
            _gamma = inj_result.gamma if inj_result else 1.0
            _regime = inj_result.regime if inj_result else "neutral"
            _alignment = inj_result.alignment if inj_result else 1.0

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=action,
                action_valid=action_valid,
                reward=reward,
                confidence=confidence,
                skills_injected=injected > 0,
                skills_used=retrieved_skill_names,
                skill_scores=retrieved_skill_scores,
                injection_strength=strength,
                drift_rate=_drift_rate,
                drift_norm=_drift_norm,
                adaptive_momentum=_adaptive_momentum,
                isolation_score=_isolation,
                in_uncharted=_uncharted,
                gamma=_gamma,
                regime=_regime,
                alignment=_alignment,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
            )
            logs.append(log)

            prev_confidence = confidence
            history.append((action, obs_new))
            if len(history) > self.max_history:
                history = history[-self.max_history:]

            observation = obs_new
            admissible_commands = admissible_new

            if reward > 0:
                success = True
            if done:
                break

        return success, logs

    def _inject_with_md_formula(self, t, confidence: float, force: bool = False,
                                drift_rate: float = 0.0, recently_used: list[int] | None = None,
                                filtered_vectors=None, filtered_indices=None):
        """MD-style injection with sphere pipeline.

        Args:
            filtered_vectors: If provided, search only these skill vectors.
            filtered_indices: Index mapping from filtered → original skill indices.

        Returns:
            (skill_text, n_injected, total_strength, result)
        """
        if self.skill_sphere is None:
            return "", 0, 0.0, None

        vectors = filtered_vectors if filtered_vectors is not None else self.skill_sphere.vectors

        # Remap recently_used to filtered index space
        remapped_used = None
        if recently_used and filtered_indices is not None:
            reverse_map = {orig: filt for filt, orig in enumerate(filtered_indices)}
            remapped_used = [reverse_map[idx] for idx in recently_used if idx in reverse_map]
        else:
            remapped_used = recently_used

        result = self.injector.decide(
            intent_point=t,
            confidence=confidence,
            skill_vectors=vectors,
            force_inject=force,
            drift_rate=drift_rate,
            recently_used=remapped_used,
        )

        if not result.should_inject:
            return "", 0, 0.0, result

        # Remap selected_indices back to original skill space
        if filtered_indices is not None:
            result.selected_indices = [filtered_indices[i] for i in result.selected_indices]

        text = self.injector.format_injected_skills(
            result,
            self.skill_sphere.skills,
            skill_vectors=self.skill_sphere.vectors,
        )

        return text, len(result.selected_indices), result.total_injection_strength, result

    def _format_general_skills(self, task_type: str, task_description: str = "") -> str:
        """Format general skills as always-available base context.

        If the sphere has many general skills (unified mode), use sphere
        geometric retrieval to select the most relevant ones instead of
        dumping all of them into the system prompt.
        """
        if self.skill_sphere is None:
            return ""

        general = self.skill_sphere.get_general_skills()
        if not general:
            return ""

        MAX_GENERAL = 15  # Above this, use geometric selection
        if len(general) > MAX_GENERAL and task_description and self.skill_sphere.encoder:
            # Unified sphere: geometrically select most relevant generals
            import torch
            query_vec = self.skill_sphere.encoder.encode_query(task_description)
            gen_indices = [idx for idx, _ in general]
            gen_vecs = torch.stack([self.skill_sphere._vectors[i] for i in gen_indices])
            sims = torch.mv(gen_vecs, query_vec)
            topk = min(MAX_GENERAL, len(gen_indices))
            _, top_local = torch.topk(sims, topk)
            selected = [(gen_indices[i], general[i][1]) for i in top_local.tolist()]
        else:
            selected = general

        parts = ["### General Principles"]
        for idx, skill in selected:
            parts.append(f"- **{skill.name}**: {skill.principle}")

        return "\n".join(parts)

    def _get_env_filtered_indices(self, allowed_categories: set[str]) -> list[int] | None:
        """Filter skill indices to keep only categories in allowed_categories.

        Returns None if all skills already belong to allowed categories (no filtering needed).
        """
        if self.skill_sphere is None:
            return None

        filtered = []
        for i, skill in enumerate(self.skill_sphere.skills):
            cat = (skill.category or "").lower()
            if cat in allowed_categories or cat == "":
                filtered.append(i)

        # No filtering needed if all skills pass
        if len(filtered) == len(self.skill_sphere.skills):
            return None
        return filtered

    # --- Embedding top-K baseline (cosine similarity, no sphere geometry) ---

    def _run_episode_embed_topk(
        self,
        task_description: str,
        task_type: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
        top_k: int = 5,
    ) -> tuple[bool, list[StepLog]]:
        """Embedding top-K baseline: simple cosine similarity retrieval.

        No sphere-adaptive threshold, no complementarity selection, no dynamic injection.
        Just encode task → find top-K nearest skills by cosine sim → inject statically.
        This isolates the value of sphere geometry from basic embedding retrieval.
        """
        if self.skill_sphere is None or self.skill_sphere.encoder is None:
            return self._run_episode_static(
                task_description, "", initial_observation,
                admissible_commands, step_fn, max_steps,
            )

        import torch

        # Encode task description
        query_vec = self.skill_sphere.encoder.encode_query(task_description)
        skill_vectors = self.skill_sphere.vectors  # (N, D)

        # Simple cosine similarity (vectors are L2-normalized, so dot product = cosine)
        if isinstance(skill_vectors, torch.Tensor):
            sims = torch.mv(skill_vectors, query_vec)
            topk_vals, topk_idxs = torch.topk(sims, min(top_k, len(sims)))
            topk_idxs = topk_idxs.tolist()
        else:
            import numpy as np
            sims = np.dot(skill_vectors, query_vec)
            topk_idxs = np.argsort(sims)[-top_k:][::-1].tolist()

        # Format selected skills
        parts = ["### Retrieved Skills (Top-K by similarity)"]
        for idx in topk_idxs:
            skill = self.skill_sphere.skills[idx]
            parts.append(f"- **{skill.name}**: {skill.principle}")
        skill_text = "\n".join(parts)

        return self._run_episode_static(
            task_description, skill_text, initial_observation,
            admissible_commands, step_fn, max_steps,
        )

    # --- SkillRL baseline (unchanged) ---

    def _run_episode_skillrl(
        self,
        task_description: str,
        task_type: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """SkillRL baseline: all general + all task-specific skills, injected once."""
        skill_text = self._get_skills_skillrl(task_type)
        return self._run_episode_static(
            task_description, skill_text, initial_observation,
            admissible_commands, step_fn, max_steps,
        )

    def _get_skills_skillrl(self, task_type: str) -> str:
        """SkillRL baseline: all general + all task-specific."""
        if self.skill_sphere is None:
            return ""
        parts = []

        general = self.skill_sphere.get_general_skills()
        if general:
            parts.append("### General Principles")
            for idx, skill in general:
                parts.append(f"- **{skill.name}**: {skill.principle}")
                if skill.when_to_apply:
                    parts.append(f"  _Apply when: {skill.when_to_apply}_")

        task_skills = self.skill_sphere.get_task_skills(task_type)
        if task_skills:
            parts.append(f"\n### {task_type.replace('_', ' ').title()} Skills")
            for idx, skill in task_skills:
                parts.append(f"- **{skill.name}**: {skill.principle}")
                if skill.when_to_apply:
                    parts.append(f"  _Apply when: {skill.when_to_apply}_")

        return "\n".join(parts)

    # --- SkillRL-fair baseline (replicates real SkillRL's retrieval) ---

    def _run_episode_skillrl_fair(
        self,
        task_description: str,
        task_type: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """SkillRL-fair baseline: top-6 general + task-specific + top-5 mistakes.

        Replicates the real SkillRL's skill injection logic:
        - general_skills[:6] (capped at 6)
        - All task_specific_skills for detected task_type
        - common_mistakes[:5] (capped at 5)
        - Injected in user prompt (not system prompt) under "## Retrieved Relevant Experience"
        """
        skill_text = self._get_skills_skillrl_fair(task_type)
        return self._run_episode_skillrl_fair_loop(
            task_description, skill_text, initial_observation,
            admissible_commands, step_fn, max_steps,
        )

    def _get_skills_skillrl_fair(self, task_type: str) -> str:
        """Format skills exactly as real SkillRL does."""
        if self.skill_sphere is None:
            return ""

        parts = []

        # General skills: capped at 6 (SkillRL uses first 6)
        general = self.skill_sphere.get_general_skills()
        if general:
            parts.append("### General Principles")
            for idx, skill in general[:6]:
                parts.append(f"- **{skill.name}**: {skill.principle}")

        # Task-specific skills: all for the detected task type
        task_skills = self.skill_sphere.get_task_skills(task_type)
        if task_skills:
            title = task_type.replace("_", " ").title()
            parts.append(f"\n### {title} Skills")
            for idx, skill in task_skills:
                parts.append(f"- **{skill.name}**: {skill.principle}")
                if skill.when_to_apply:
                    parts.append(f"  _Apply when: {skill.when_to_apply}_")

        # Common mistakes: capped at 5 (SkillRL uses first 5)
        mistakes = self.skill_sphere.get_common_mistakes()
        if mistakes:
            parts.append("\n### Mistakes to Avoid")
            for idx, skill in mistakes[:5]:
                # Format as SkillRL does: description + how_to_avoid
                desc = skill.principle or skill.name
                avoid = skill.when_to_apply or ""
                parts.append(f"- **Don't**: {desc}")
                if avoid:
                    parts.append(f"  **Instead**: {avoid}")

        return "\n".join(parts)

    def _run_episode_skillrl_fair_loop(
        self,
        task_description: str,
        skill_text: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """Run episode with SkillRL-fair injection (skills in user prompt, not system)."""
        history: list[tuple[str, str]] = []
        logs: list[StepLog] = []
        observation = initial_observation
        success = False

        for step_num in range(1, max_steps + 1):
            # System prompt: base only (no skills — SkillRL puts skills in user msg)
            system_prompt = SYSTEM_PROMPT_BASE

            history_block = self._format_history(history)
            admissible_str = ", ".join(f"'{a}'" for a in admissible_commands)

            # SkillRL-style: skills go in user prompt under "Retrieved Relevant Experience"
            skill_section = ""
            if skill_text:
                skill_section = f"\n\n## Retrieved Relevant Experience\n\n{skill_text}\n"

            user_prompt = STEP_PROMPT_TEMPLATE.format(
                task_description=task_description,
                history_block=history_block,
                step_num=step_num,
                observation=observation,
                admissible_actions=admissible_str,
            ) + skill_section

            raw_output = self.llm.generate_action(system_prompt, user_prompt)
            thinking, action, action_valid = parse_action(raw_output)
            action = match_to_admissible(action, admissible_commands)

            obs_new, reward, done, admissible_new = step_fn(action)

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=action,
                action_valid=action_valid,
                reward=reward,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
            )
            logs.append(log)

            history.append((action, obs_new))
            if len(history) > self.max_history:
                history = history[-self.max_history:]

            observation = obs_new
            admissible_commands = admissible_new

            if reward > 0:
                success = True
            if done:
                break

        return success, logs

    # --- SkillRL-native mode (exact GRPO training format) ---

    def _run_episode_skillrl_native(
        self,
        task_description: str,
        task_type: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
        use_sphere: bool = False,
    ) -> tuple[bool, list[StepLog]]:
        """Run with SkillRL's exact prompt template + skill retrieval.

        Uses single-message format matching SkillRL's GRPO training template.
        use_sphere=False: fair (template) retrieval
        use_sphere=True: full DynamicInjector with per-step re-retrieval
        """
        import torch
        from skill_sphere.injection.dynamic_inject import DynamicInjector
        from skill_sphere.injection.intent_tracker import IntentTracker

        # Setup sphere injection if available
        use_dynamic = (use_sphere and self.skill_sphere is not None
                       and self.skill_sphere.encoder is not None)
        injector = None
        intent_tracker = None
        skill_vectors = None
        recently_used = []

        if use_dynamic:
            encoder = self.skill_sphere.encoder
            skill_vectors = self.skill_sphere.vectors
            injector = DynamicInjector(
                sigma=1.5, min_inject_strength=0.15, max_skills=5, relevance_k=10,
            )
            injector.calibrate(skill_vectors)
            intent_tracker = IntentTracker()
            intent_tracker.calibrate(injector.retriever.d_typical)
            with torch.no_grad():
                initial_intent = encoder.encode_query(task_description)
            intent_tracker.reset(initial_intent)

        if not use_dynamic and not use_sphere:
            skill_text = self._get_skills_skillrl_fair(task_type)
        else:
            skill_text = ""  # Will be set per-step below

        history: list[tuple[str, str]] = []
        logs: list[StepLog] = []
        observation = initial_observation
        success = False

        for step_num in range(1, max_steps + 1):
            # Per-step dynamic skill retrieval
            if use_dynamic:
                context_text = f"{task_description}. Step {step_num}: {observation[:300]}"
                with torch.no_grad():
                    intent_point = encoder.encode_query(context_text)
                drift_info = intent_tracker.update(intent_point)
                drift_rate = drift_info.drift_rate if drift_info else 0.0

                injection_result = injector.decide(
                    intent_point=intent_point,
                    confidence=0.3,
                    skill_vectors=skill_vectors,
                    force_inject=True,
                    drift_rate=drift_rate,
                    recently_used=recently_used[-3:] if recently_used else None,
                )
                if injection_result.should_inject:
                    skill_text = injector.format_injected_skills(
                        injection_result, self.skill_sphere.skills, skill_vectors,
                    )
                    recently_used.extend(injection_result.selected_indices)
                else:
                    skill_text = ""

            admissible_str = ", ".join(f"'{a}'" for a in admissible_commands)

            if step_num == 1:
                # First step: no history
                if skill_text:
                    user_prompt = SKILLRL_NATIVE_TEMPLATE_WITH_MEMORY.format(
                        task_description=task_description,
                        retrieved_memories=skill_text,
                        step_count=0,
                        history_length=0,
                        action_history="None",
                        current_step=step_num,
                        current_observation=observation,
                        admissible_actions=admissible_str,
                    )
                else:
                    user_prompt = SKILLRL_NATIVE_TEMPLATE_NO_HIS.format(
                        current_observation=observation,
                        admissible_actions=admissible_str,
                    )
            else:
                # Format history as SkillRL does
                history_strs = []
                recent = history[-self.max_history:]
                for i, (act, obs) in enumerate(recent):
                    history_strs.append(
                        f"\nStep {step_num - len(recent) + i}: "
                        f"Action: {act} | Observation: {obs}"
                    )
                action_history = "".join(history_strs) if history_strs else "None"

                if skill_text:
                    user_prompt = SKILLRL_NATIVE_TEMPLATE_WITH_MEMORY.format(
                        task_description=task_description,
                        retrieved_memories=skill_text,
                        step_count=step_num - 1,
                        history_length=len(recent),
                        action_history=action_history,
                        current_step=step_num,
                        current_observation=observation,
                        admissible_actions=admissible_str,
                    )
                else:
                    user_prompt = SKILLRL_NATIVE_TEMPLATE.format(
                        task_description=task_description,
                        step_count=step_num - 1,
                        history_length=len(recent),
                        action_history=action_history,
                        current_step=step_num,
                        current_observation=observation,
                        admissible_actions=admissible_str,
                    )

            # Single-message: no system prompt, everything in user message
            raw_output = self.llm.generate_action("", user_prompt)
            thinking, action, action_valid = parse_action(raw_output)
            action = match_to_admissible(action, admissible_commands)

            obs_new, reward, done, admissible_new = step_fn(action)

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=action,
                action_valid=action_valid,
                reward=reward,
                system_prompt="",
                user_prompt=user_prompt,
                raw_output=raw_output,
            )
            logs.append(log)

            history.append((action, obs_new))
            observation = obs_new
            admissible_commands = admissible_new

            if reward > 0:
                success = True
            if done:
                break

        return success, logs

    # --- Static mode (none / skillrl) ---

    def _run_episode_static(
        self,
        task_description: str,
        skill_text: str,
        initial_observation: str,
        admissible_commands: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """Run episode with static skill injection (none or skillrl mode)."""
        history: list[tuple[str, str]] = []
        logs: list[StepLog] = []
        observation = initial_observation
        success = False

        for step_num in range(1, max_steps + 1):
            system_prompt = SYSTEM_PROMPT_BASE
            if skill_text:
                system_prompt += SKILL_INJECTION_TEMPLATE.format(
                    skill_text=skill_text,
                    confidence_note="",
                )

            history_block = self._format_history(history)
            admissible_str = ", ".join(f"'{a}'" for a in admissible_commands)
            user_prompt = STEP_PROMPT_TEMPLATE.format(
                task_description=task_description,
                history_block=history_block,
                step_num=step_num,
                observation=observation,
                admissible_actions=admissible_str,
            )

            raw_output = self.llm.generate_action(system_prompt, user_prompt)
            thinking, action, action_valid = parse_action(raw_output)
            action = match_to_admissible(action, admissible_commands)

            obs_new, reward, done, admissible_new = step_fn(action)

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=action,
                action_valid=action_valid,
                reward=reward,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
            )
            logs.append(log)

            history.append((action, obs_new))
            if len(history) > self.max_history:
                history = history[-self.max_history:]

            observation = obs_new
            admissible_commands = admissible_new

            if reward > 0:
                success = True
            if done:
                break

        return success, logs

    def _format_history(self, history: list[tuple[str, str]]) -> str:
        """Format action-observation history for the prompt."""
        if not history:
            return ""

        lines = [f"Previous steps ({len(history)} most recent):"]
        for i, (action, obs) in enumerate(history):
            lines.append(f"  Action: {action}")
            lines.append(f"  Result: {obs}")
        return "\n".join(lines)
