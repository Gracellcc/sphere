"""AppWorld agent with Skill Sphere retrieval.

Implements the Skill Sphere inference loop for AppWorld:
1. Maintain a policy intent point `t` on the sphere that evolves per step
2. Confidence-guided dynamic injection (inject when uncertain)
3. Per-step retrieval based on current API context
4. Slerp combination + complementarity selection

Supports modes: "sphere", "embed_topk", "none"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from skill_sphere.agent.llm_client import LLMClient
from skill_sphere.skill_bank.skill_sphere import SkillSphere
from skill_sphere.geometry.sphere import slerp
from skill_sphere.injection.dynamic_inject import DynamicInjector
from skill_sphere.injection.confidence import compute_logit_confidence, SphereConfidence
from skill_sphere.injection.intent_tracker import IntentTracker


# --- Action parsing ---

def _looks_like_python(text: str) -> bool:
    """Heuristic check if text looks like Python code (not English text)."""
    first_line = text.strip().split("\n")[0].strip()
    # Must start with something code-like, not English prose
    code_starters = ["apis.", "print(", "import ", "from ", "token", "result", "#", "for ", "if ", "while "]
    fl = first_line.lower()
    return any(fl.startswith(s) for s in code_starters) or "= apis." in fl


def parse_code_action(raw_output: str) -> tuple[str, str, bool]:
    """Parse agent output to extract thinking and code action.

    Returns:
        (thinking, code, found_code_tag)
    """
    # Extract thinking
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Try <code> tags first
    code_match = re.search(r"<code>(.*?)</code>", raw_output, re.DOTALL)
    if code_match:
        return thinking, code_match.group(1).strip(), True

    # Try markdown code blocks
    md_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if md_match:
        return thinking, md_match.group(1).strip(), True

    # Fallback: if output has </think> followed by Python-like code
    think_end = raw_output.find("</think>")
    if think_end >= 0:
        after_think = raw_output[think_end + len("</think>"):].strip()
        # Strip repeated </think> tags (model sometimes outputs multiple)
        while after_think.startswith("</think>"):
            after_think = after_think[len("</think>"):].strip()
        if after_think and _looks_like_python(after_think):
            return thinking, after_think.strip(), True

    # Last resort: if remaining text (after thinking) looks like Python
    remaining = raw_output
    if think_match:
        remaining = raw_output[think_match.end():].strip()
    if remaining and _looks_like_python(remaining):
        return thinking, remaining.strip(), True

    return thinking, "", False


def detect_task_complete(code: str) -> bool:
    """Check if the code signals task completion."""
    return "complete_task" in code or "TASK_COMPLETE" in code


# --- Confidence estimation ---

def estimate_confidence(raw_output: str) -> float:
    """Estimate confidence from agent output."""
    confidence = 0.5

    has_think = "<think>" in raw_output and "</think>" in raw_output
    has_code = "<code>" in raw_output and "</code>" in raw_output
    if has_think and has_code:
        confidence += 0.2

    think_text = ""
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).lower()

    hedging = ["not sure", "maybe", "try", "perhaps", "might", "unclear",
               "don't know", "error", "failed", "traceback"]
    for word in hedging:
        if word in think_text:
            confidence -= 0.1
            break

    confident = ["i know", "clearly", "simple", "straightforward", "just need to"]
    for word in confident:
        if word in think_text:
            confidence += 0.1
            break

    return max(0.0, min(1.0, confidence))


# --- Prompt templates ---

SYSTEM_PROMPT_BASE = """You are an AI Assistant whose job is to complete day-to-day tasks fully autonomously on behalf of a supervisor.

To do this, you interact with apps (e.g., spotify, venmo, gmail, etc.) using their APIs. You undertake a multi-step conversation using a Python REPL environment. You write Python code, the environment executes it and shows you the result, based on which you write the next step, and so on until the goal is achieved.

## How to discover and use APIs

There are 3 key APIs for discovering available functionality:
1. `apis.api_docs.show_app_descriptions()` — lists all available apps with descriptions
2. `apis.api_docs.show_api_descriptions(app_name="APP")` — lists all APIs for a given app
3. `apis.api_docs.show_api_doc(app_name="APP", api_name="API")` — shows full spec for an API (parameters, types, constraints, response schema)

Always check the API spec before calling an unfamiliar API.

## How to log in to apps

Most APIs require an access_token. To get one:
1. Get passwords: `print(apis.supervisor.show_account_passwords())`
   This returns a dict like {"spotify": {"username": "...", "password": "..."}, ...}
2. Log in: `result = apis.APP.login(username=USERNAME, password=PASSWORD)`
   The result contains an `access_token` field.
3. Use the token: `apis.APP.some_api(access_token=token, ...)`

Username is usually the supervisor's email. For the phone app, it's the phone number.

## How to handle pagination

Many APIs return paginated results. Always loop through all pages:
```python
page = 1
all_items = []
while True:
    result = apis.APP.some_list_api(access_token=token, page=page)
    all_items.extend(result)
    if len(result) == 0:
        break
    page += 1
```

## How to complete the task

When done, call `apis.supervisor.complete_task(...)`:
- For question tasks (asking for info): `apis.supervisor.complete_task(answer=YOUR_ANSWER, status="success")`
  - Keep answers minimal: just the entity/number, no extra text
  - Use numeric format (e.g., "3" not "three")
- For action tasks (do something): `apis.supervisor.complete_task(status="success")`
- If stuck: `apis.supervisor.complete_task(status="fail")`

## Key rules

A. General:
- Act fully autonomously. Never ask for clarification.
- Never invent or guess values — always look them up via APIs.
- Avoid collateral damage — only do what's explicitly asked.
- If a detail is omitted from the task, pick any valid value.

B. App-specific:
- Personal info (name, email, phone, addresses, payment cards) is in the Supervisor app.
- Friends/family info = phone contacts.
- Get current date/time from `datetime.datetime.now()` or `apis.phone.show_current_datetime()`.
- "file system" means the file_system app, not the OS.

C. Code:
- Variables persist across code blocks.
- Write small chunks of code, one step at a time.
- Always use `print()` to see API return values.
- Only use standard library + the provided `apis` object. No external packages.
- Do NOT import apis or appworld modules. `apis` is pre-loaded.

Wrap your code in ```python ... ``` blocks.
"""

SKILL_INJECTION_TEMPLATE = """
## Skill Guidance{confidence_note}

{skill_text}"""

FIRST_USER_TEMPLATE = """My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

Task: {task_description}"""

OUTPUT_TEMPLATE = """Output:
```
{observation}
```"""

# Max chars per observation in chat history to prevent context overflow
MAX_OBS_CHARS = 2000


def truncate_obs(obs: str, max_chars: int = MAX_OBS_CHARS) -> str:
    """Truncate long API outputs to keep chat history manageable."""
    if len(obs) <= max_chars:
        return obs
    head = max_chars * 2 // 3
    tail = max_chars // 3
    return obs[:head] + f"\n... [truncated {len(obs) - head - tail} chars] ...\n" + obs[-tail:]


# --- Agent ---

@dataclass
class StepLog:
    """Log for a single agent step."""
    step: int
    observation: str
    thinking: str
    action: str  # The code that was executed
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
    # Full prompt/response for trajectory recording
    system_prompt: str = ""
    user_prompt: str = ""
    raw_output: str = ""


class AppWorldAgent:
    """Agent that uses Skill Sphere for AppWorld tasks."""

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
        self.llm = llm
        self.skill_sphere = skill_sphere
        self.mode = mode
        self.max_history = max_history
        self.intent_momentum = intent_momentum
        self.min_inject_strength = min_inject_strength
        self.confidence_temperature = confidence_temperature

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

        self.intent_tracker = IntentTracker()
        self.sgc = SphereConfidence()

    def run_episode(
        self,
        task_description: str,
        supervisor_info: dict,
        allowed_apps: list[str],
        step_fn,
        max_steps: int = 30,
    ) -> tuple[bool, list[StepLog]]:
        """Run a complete AppWorld episode.

        Args:
            task_description: The task instruction.
            supervisor_info: Supervisor details (name, email, etc.).
            allowed_apps: List of available app names.
            step_fn: Callable(code) -> (observation, task_completed)
            max_steps: Maximum number of interactions.

        Returns:
            (success, logs) where success=True if task was completed.
        """
        if self.mode == "embed_topk":
            return self._run_episode_embed_topk(
                task_description, supervisor_info, allowed_apps,
                step_fn, max_steps,
            )
        elif self.mode == "none":
            return self._run_episode_static(
                task_description, "", supervisor_info, allowed_apps,
                step_fn, max_steps,
            )
        else:  # sphere
            return self._run_episode_sphere(
                task_description, supervisor_info, allowed_apps,
                step_fn, max_steps,
            )

    def _run_episode_sphere(
        self,
        task_description: str,
        supervisor_info: dict,
        allowed_apps: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """Full Skill Sphere inference loop for AppWorld."""
        if self.skill_sphere is None or self.skill_sphere.encoder is None:
            return self._run_episode_static(
                task_description, "", supervisor_info, allowed_apps,
                step_fn, max_steps,
            )

        # Initialize intent point from task description
        t = self.skill_sphere.encoder.encode_query(task_description)

        # Calibrate sphere components
        self.injector.calibrate(self.skill_sphere.vectors)
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

        # --- App-aware skill filtering ---
        # Detect which apps the task actually needs from its description
        # (Like ALFWorld's task_type filtering, but for AppWorld's app categories)
        detected_apps = self._detect_relevant_apps(task_description)
        filtered_indices = self._get_filtered_skill_indices(detected_apps)

        if filtered_indices and len(filtered_indices) < len(self.skill_sphere.skills):
            filtered_vectors = torch.stack([self.skill_sphere.vectors[i] for i in filtered_indices])
        else:
            filtered_vectors = None
            filtered_indices = None

        # General skills always available
        general_text = self._format_general_skills(task_description)

        logs: list[StepLog] = []
        observation = ""
        success = False
        prev_drift: float = 0.0
        prev_coherence: float = 1.0
        prev_stability: float = 1.0
        prev_codes: list[str] = []
        recently_used_skills: list[int] = []

        # Build system prompt with general skills
        system_prompt = SYSTEM_PROMPT_BASE
        if general_text:
            system_prompt += SKILL_INJECTION_TEMPLATE.format(
                skill_text=general_text,
                confidence_note=" (General Principles)",
            )

        # First user message (SRPO style)
        first_user = FIRST_USER_TEMPLATE.format(
            first_name=supervisor_info.get("first_name", ""),
            last_name=supervisor_info.get("last_name", ""),
            email=supervisor_info.get("email", ""),
            phone_number=supervisor_info.get("phone_number", ""),
            task_description=task_description,
        )

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]

        for step_num in range(1, max_steps + 1):
            user_prompt = chat_messages[-1]["content"]

            # --- Sphere Injection ---
            in_loop = (
                len(prev_codes) >= 2
                and prev_codes[-1] == prev_codes[-2]
            )
            force = in_loop or step_num == 1

            isolation = self.injector.compute_isolation(t, self.skill_sphere.vectors)
            sgc_signals = self.sgc.compute(
                coherence=prev_coherence,
                stability=prev_stability,
                isolation_score=isolation,
            )
            effective_confidence = 0.1 if in_loop else sgc_signals.sgc

            rotation_active = in_loop or effective_confidence < 0.4
            rotation_list = recently_used_skills if rotation_active else None

            skill_text, injected, strength, inj_result = self._inject_with_md_formula(
                t, effective_confidence, force=force, drift_rate=prev_drift,
                recently_used=rotation_list,
                filtered_vectors=filtered_vectors,
                filtered_indices=filtered_indices,
            )

            if inj_result and inj_result.in_uncharted and not inj_result.should_inject:
                skill_text = general_text if general_text else ""

            # Inject skills into the last user message
            if skill_text:
                if in_loop:
                    skill_text += "\n\nIMPORTANT: You seem to be repeating the same code. Try a DIFFERENT approach."
                conf_note = f" (confidence={effective_confidence:.2f}, w={strength:.2f})"
                injection = SKILL_INJECTION_TEMPLATE.format(
                    skill_text=skill_text,
                    confidence_note=conf_note,
                )
                # Temporarily append to last user message for this generation
                injected_messages = chat_messages[:-1] + [
                    {"role": "user", "content": chat_messages[-1]["content"] + injection}
                ]
            else:
                injected_messages = chat_messages

            raw_output = self.llm.generate_chat(injected_messages)
            thinking, code, code_valid = parse_code_action(raw_output)
            confidence = effective_confidence

            if not code:
                chat_messages.append({"role": "assistant", "content": raw_output})
                chat_messages.append({"role": "user", "content": "Please write Python code to execute. Wrap it in ```python ... ``` blocks."})
                code = 'print("No code generated")'
                code_valid = False

            prev_codes.append(code)

            obs_new, task_completed = step_fn(code)

            # Update intent point
            context_text = f"{task_description}. Step {step_num}: {code[:100]}. Result: {obs_new[:200]}"
            drift_info = None
            try:
                context_vec = self.skill_sphere.encoder.encode_query(context_text)
                drift_info = self.intent_tracker.update(context_vec)
                t = slerp(
                    t.unsqueeze(0),
                    context_vec.unsqueeze(0),
                    drift_info.alpha,
                ).squeeze(0)
                prev_drift = drift_info.drift_rate
                prev_coherence = drift_info.coherence
                prev_stability = drift_info.stability
            except Exception:
                prev_drift = 0.0

            if inj_result and inj_result.selected_indices:
                recently_used_skills.extend(inj_result.selected_indices)
                recently_used_skills = recently_used_skills[-15:]

            retrieved_skill_names = []
            retrieved_skill_scores = []
            if inj_result and inj_result.selected_indices:
                for idx, score in zip(inj_result.selected_indices, inj_result.injection_weights):
                    retrieved_skill_names.append(self.skill_sphere.skills[idx].name)
                    retrieved_skill_scores.append(score)

            _drift_rate = drift_info.drift_rate if drift_info else prev_drift
            _drift_norm = drift_info.drift_norm if drift_info else 0.0
            _adaptive_momentum = drift_info.alpha if drift_info else 0.0
            _isolation = inj_result.isolation_score if inj_result else 0.0
            _uncharted = inj_result.in_uncharted if inj_result else False
            _gamma = inj_result.gamma if inj_result else 1.0
            _regime = inj_result.regime if inj_result else "neutral"
            _alignment = inj_result.alignment if inj_result else 1.0

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=code,
                action_valid=code_valid,
                reward=1.0 if task_completed else 0.0,
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

            # Append to chat (SRPO style)
            if code_valid:
                chat_messages.append({"role": "assistant", "content": raw_output})
                chat_messages.append({"role": "user", "content": OUTPUT_TEMPLATE.format(observation=truncate_obs(obs_new))})

            observation = obs_new

            if task_completed:
                success = True
                break

        return success, logs

    def _inject_with_md_formula(self, t, confidence: float, force: bool = False,
                                drift_rate: float = 0.0, recently_used: list[int] | None = None,
                                filtered_vectors=None, filtered_indices=None):
        """MD-style injection with sphere pipeline.

        Args:
            filtered_vectors: If provided, search only these skill vectors.
            filtered_indices: Index mapping from filtered → original skill indices.
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

    def _format_general_skills(self, task_description: str = "") -> str:
        """Format general skills as always-available base context.

        If the sphere has many general skills (unified mode), use geometric
        selection to pick the most relevant ones.
        """
        if self.skill_sphere is None:
            return ""

        general = self.skill_sphere.get_general_skills()
        if not general:
            return ""

        MAX_GENERAL = 15
        if len(general) > MAX_GENERAL and task_description and self.skill_sphere.encoder:
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

    def _format_common_mistakes(self) -> str:
        """Format common mistakes as static context."""
        if self.skill_sphere is None:
            return ""

        mistakes = self.skill_sphere.get_common_mistakes()
        if not mistakes:
            return ""

        parts = ["### Common Mistakes to Avoid"]
        for idx, skill in mistakes:
            # Mistakes have description/how_to_avoid in principle field
            parts.append(f"- {skill.principle}")

        return "\n".join(parts)

    @staticmethod
    def _detect_relevant_apps(task_description: str) -> list[str]:
        """Detect which apps are relevant to a task from its description.

        Uses keyword matching with both explicit app names and implicit
        action/object references (e.g. "liked songs" → spotify).
        """
        desc = task_description.lower()
        app_keywords = {
            "spotify": ["spotify", "song", "playlist", "album", "music", "artist",
                         "queue", "player", "listen", "liked song", "shuffle",
                         "recommended song", "next song", "previous song"],
            "venmo": ["venmo", "payment request", "befriend", "friends on venmo",
                       "venmo friend", "transaction", "pay request"],
            "gmail": ["gmail", "email", "mail", "inbox", "sent mail", "draft"],
            "phone": ["phone", "text message", "alarm", "contact", "call",
                       "voice message", "sms", "my phone"],
            "file_system": ["file", "directory", "folder", "download", "~/",
                            "path", ".jpg", ".png", ".csv", ".txt", ".md",
                            "markdown", "export", "import"],
            "simple_note": ["simple note", "simplenote", "note", "habit",
                            "log note", "tracking log", "note in my"],
            "todoist": ["todoist", "todo", "to-do", "project task"],
            "splitwise": ["splitwise", "expense", "split the", " owe ", "i owe", "they owe",
                          "group expense", "dinner", "trip with", "taxi", "paid for"],
            "amazon": ["amazon", "product", "order", "cart", "buy", "purchase",
                        "review", "rating", "wishlist"],
        }

        detected = []
        for app, keywords in app_keywords.items():
            if any(kw in desc for kw in keywords):
                detected.append(app)

        # Multi-app detection: if 2+ apps detected, also include multi_app category
        if len(detected) >= 2:
            detected.append("multi_app")

        return detected

    def _get_filtered_skill_indices(self, relevant_apps: list[str]) -> list[int]:
        """Get indices of skills relevant to detected apps + general + mistakes."""
        if self.skill_sphere is None:
            return []

        indices = set()

        # Always include general skills and common mistakes
        for idx, _ in self.skill_sphere.get_general_skills():
            indices.add(idx)
        for idx, _ in self.skill_sphere.get_common_mistakes():
            indices.add(idx)

        # Include skills for detected apps
        for app in relevant_apps:
            for idx, _ in self.skill_sphere.get_task_skills(app):
                indices.add(idx)

        # Also include multi_app skills (useful for cross-app tasks)
        for idx, _ in self.skill_sphere.get_task_skills("multi_app"):
            indices.add(idx)

        # If no apps detected, include all skills
        if not relevant_apps:
            return list(range(len(self.skill_sphere.skills)))

        return sorted(indices)

    # --- Embedding top-K baseline ---

    def _run_episode_embed_topk(
        self,
        task_description: str,
        supervisor_info: dict,
        allowed_apps: list[str],
        step_fn,
        max_steps: int,
        top_k: int = 5,
    ) -> tuple[bool, list[StepLog]]:
        """Embedding top-K baseline: cosine similarity, static injection."""
        if self.skill_sphere is None or self.skill_sphere.encoder is None:
            return self._run_episode_static(
                task_description, "", supervisor_info, allowed_apps,
                step_fn, max_steps,
            )

        import torch

        query_vec = self.skill_sphere.encoder.encode_query(task_description)

        # App-aware filtering (same as sphere mode for fair comparison)
        detected_apps = self._detect_relevant_apps(task_description)
        filtered_idx = self._get_filtered_skill_indices(detected_apps)
        if filtered_idx and len(filtered_idx) < len(self.skill_sphere.skills):
            skill_vectors = torch.stack([self.skill_sphere.vectors[i] for i in filtered_idx])
            idx_map = filtered_idx  # maps filtered position → original index
        else:
            skill_vectors = self.skill_sphere.vectors
            idx_map = None

        sims = torch.mv(skill_vectors, query_vec)
        topk_vals, topk_idxs = torch.topk(sims, min(top_k, len(sims)))

        parts = ["### Retrieved Skills (Top-K by similarity)"]
        for local_idx in topk_idxs.tolist():
            idx = idx_map[local_idx] if idx_map else local_idx
            skill = self.skill_sphere.skills[idx]
            parts.append(f"- **{skill.name}**: {skill.principle}")
        skill_text = "\n".join(parts)

        return self._run_episode_static(
            task_description, skill_text, supervisor_info, allowed_apps,
            step_fn, max_steps,
        )

    # --- Static mode (none / embed_topk with pre-formatted skills) ---

    def _run_episode_static(
        self,
        task_description: str,
        skill_text: str,
        supervisor_info: dict,
        allowed_apps: list[str],
        step_fn,
        max_steps: int,
    ) -> tuple[bool, list[StepLog]]:
        """Run episode with SRPO-style multi-turn chat format."""
        logs: list[StepLog] = []
        observation = ""
        success = False

        # Build system prompt (with optional skill injection)
        system_prompt = SYSTEM_PROMPT_BASE
        if skill_text:
            system_prompt += SKILL_INJECTION_TEMPLATE.format(
                skill_text=skill_text,
                confidence_note="",
            )

        # First user message: supervisor info + task
        first_user = FIRST_USER_TEMPLATE.format(
            first_name=supervisor_info.get("first_name", ""),
            last_name=supervisor_info.get("last_name", ""),
            email=supervisor_info.get("email", ""),
            phone_number=supervisor_info.get("phone_number", ""),
            task_description=task_description,
        )

        # Multi-turn chat history (accumulated messages)
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]

        for step_num in range(1, max_steps + 1):
            # Current user prompt (for logging)
            user_prompt = chat_messages[-1]["content"]

            raw_output = self.llm.generate_chat(chat_messages)
            thinking, code, code_valid = parse_code_action(raw_output)

            if not code:
                # Ask model to produce code (same as SRPO)
                chat_messages.append({"role": "assistant", "content": raw_output})
                chat_messages.append({"role": "user", "content": "Please write Python code to execute. Wrap it in ```python ... ``` blocks."})
                code = 'print("No code generated")'
                code_valid = False

            obs_new, task_completed = step_fn(code)

            log = StepLog(
                step=step_num,
                observation=observation,
                thinking=thinking,
                action=code,
                action_valid=code_valid,
                reward=1.0 if task_completed else 0.0,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
            )
            logs.append(log)

            # Append to chat history (SRPO style)
            if code_valid:
                chat_messages.append({"role": "assistant", "content": raw_output})
                chat_messages.append({"role": "user", "content": OUTPUT_TEMPLATE.format(observation=truncate_obs(obs_new))})

            observation = obs_new

            if task_completed:
                success = True
                break

        return success, logs

    @staticmethod
    def _format_supervisor(supervisor_info: dict) -> str:
        """Format supervisor info for the prompt."""
        parts = []
        if supervisor_info.get("first_name"):
            parts.append(f"{supervisor_info['first_name']} {supervisor_info.get('last_name', '')}")
        if supervisor_info.get("email"):
            parts.append(f"email: {supervisor_info['email']}")
        if supervisor_info.get("phone_number"):
            parts.append(f"phone: {supervisor_info['phone_number']}")
        return ", ".join(parts) if parts else "Unknown supervisor"

    @staticmethod
    def _format_login_credentials(supervisor_info: dict, allowed_apps: list[str]) -> str:
        """Format per-app login credentials for the prompt."""
        account_passwords = supervisor_info.get("account_passwords", {})
        email = supervisor_info.get("email", "")
        phone = supervisor_info.get("phone_number", "")

        # Filter to non-system apps that have passwords
        skip = {"api_docs", "supervisor"}
        lines = []
        for app in allowed_apps:
            if app in skip:
                continue
            pw = account_passwords.get(app, "")
            if not pw:
                continue
            username = phone if app == "phone" else email
            lines.append(f"  apis.{app}.login(username='{username}', password='{pw}')")

        if not lines:
            return "  (No credentials available — check apis.supervisor.show_profile())"
        return "\n".join(lines)

    def _format_history(self, history: list[tuple[str, str]]) -> str:
        """Format code-observation history for the prompt."""
        if not history:
            return ""

        lines = [f"Previous steps ({len(history)} most recent):"]
        for i, (code, obs) in enumerate(history):
            lines.append(f"  Code: {code}")
            lines.append(f"  Output: {obs}")
        return "\n".join(lines)
