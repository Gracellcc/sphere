"""
ATS Outer Evolution Loop for veRL GRPO training.

Implements the design doc's outer loop:
  Every M inner steps:
    1. Collect diagnostics (training stats + sphere spatial info)
    2. Generate G candidate skill sets via API (Phase 1) or model (Phase 2)
    3. Proxy eval each candidate on small task batch
    4. Select best candidate → update active skills
    5. Re-encode modified skills → sphere auto-updates

Behavioral skills: updated every M steps
Training skills: updated every K*M steps (K>1, e.g. K=3)

Usage:
    # Standalone (called between veRL epochs):
    python scripts/verl/ats_outer_loop.py \
        --skills_path data/skills/appworld_skills_ats.json \
        --training_skills_path data/skills/training_skills.json \
        --trajectory_dir results/ats_grpo/ \
        --step 100

    # Or imported and called from a veRL callback.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from openai import AzureOpenAI


class ATSOuterLoop:
    """Outer evolution loop: diagnostics → candidates → proxy eval → update.

    Design doc mapping:
    - Phase 1: API generates candidates (GPT-5.4)
    - Phase 2: Model generates candidates (future)
    """

    def __init__(
        self,
        skills_path: str,
        training_skills_path: str,
        trajectory_dir: str,
        # Evolution config
        M: int = 10,                # behavioral update frequency (every M inner steps)
        K: int = 3,                 # training update = every K*M steps
        G: int = 3,                 # number of candidates per evolution step
        proxy_eval_tasks: int = 10, # tasks for proxy evaluation
        # API config
        verifier_model: str = "gpt-5.4",
        azure_endpoint: str = None,
        azure_api_key: str = None,
        # Sphere config
        sphere_path: str = None,
        # Proxy eval / Phase 2 config
        proxy_model_path: str = None,  # model path for proxy eval vLLM
        model_candidate_ratio: float = 0.0,  # 0.0=all API (Phase 1), 1.0=all model (Phase 2)
        # Output
        output_dir: str = "results/ats_evolution",
    ):
        self.skills_path = skills_path
        self.training_skills_path = training_skills_path
        self.trajectory_dir = trajectory_dir
        self.M = M
        self.K = K
        self.G = G
        self.proxy_eval_tasks = proxy_eval_tasks
        self.proxy_model_path = proxy_model_path
        self.model_candidate_ratio = model_candidate_ratio
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load current skills
        self.behavioral_skills = self._load_json(skills_path)
        self.training_skills = self._load_json(training_skills_path)
        self._active_skill_id = self._load_active_skill_id(training_skills_path)
        self.active_training_skill = self._find_active_training_skill()

        # API client for candidate generation
        _endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not _endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT must be set (via argument or environment variable)"
            )
        self.client = AzureOpenAI(
            api_key=azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=_endpoint,
            api_version="2025-01-01-preview",
        )
        self.verifier_model = verifier_model

        # Sphere (optional, for spatial diagnostics)
        self.sphere = None
        if sphere_path:
            try:
                from skill_sphere.skill_bank.skill_sphere import SkillSphere
                self.sphere = SkillSphere.from_skillrl_json(sphere_path)
            except Exception as e:
                print(f"[ATS Outer] Could not load sphere: {e}")

        # Evolution history
        self.history = []

    def _load_json(self, path: str) -> List[Dict]:
        if not path or not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Handle wrapped formats: {"training_skills": [...]} or {"skills": [...]}
            for key in ("training_skills", "skills", "behavioral_skills"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Handle nested structure (general_skills, task_specific_skills, common_mistakes)
            # Same recursive collect as reward_manager_ats._load_behavioral_skills
            skills = []
            self._collect_skills(data, skills)
            if skills:
                return skills
            # Single dict → wrap in list
            return [data]
        return []

    @staticmethod
    def _collect_skills(obj, out: list):
        """Recursively collect skill dicts from nested structure."""
        if isinstance(obj, list):
            for item in obj:
                ATSOuterLoop._collect_skills(item, out)
        elif isinstance(obj, dict):
            if "title" in obj or "scoring" in obj:
                if obj.get("active", True):
                    out.append(obj)
            else:
                for v in obj.values():
                    ATSOuterLoop._collect_skills(v, out)

    def _load_active_skill_id(self, path: str) -> Optional[str]:
        """Read active_skill_id from training skills file (if present)."""
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("active_skill_id")
        return None

    def _find_active_training_skill(self) -> Optional[Dict]:
        # 1. Match by active_skill_id from file
        if self._active_skill_id:
            for s in self.training_skills:
                if s.get("skill_id") == self._active_skill_id:
                    return s
        # 2. Explicit active=True flag (set by evolution)
        for s in self.training_skills:
            if s.get("active", False):
                return s
        # 3. Fallback to first
        return self.training_skills[0] if self.training_skills else None

    # ── Step 1: Diagnostics ──────────────────────────────────────────────

    def collect_diagnostics(self, step: int) -> Dict[str, Any]:
        """Collect training stats + sphere spatial info.

        Reads recent trajectories from trajectory_dir.
        """
        diagnostics = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_stats": {},
            "sphere_info": {},
        }

        # Load recent trajectories
        trajectories = self._load_recent_trajectories(step)
        if not trajectories:
            print(f"[ATS Outer] No trajectories found for step {step}")
            return diagnostics

        # Training stats
        stats = self._compute_training_stats(trajectories)
        diagnostics["training_stats"] = stats

        # Sphere spatial info (if sphere available)
        if self.sphere:
            spatial = self._compute_sphere_diagnostics()
            diagnostics["sphere_info"] = spatial

        # Skill score distributions
        skill_scores = self._compute_skill_scores(trajectories)
        diagnostics["skill_scores"] = skill_scores

        return diagnostics

    def _load_recent_trajectories(self, step: int) -> List[Dict]:
        """Load recent trajectory files from trajectory_dir.

        Only loads the most recent trajectory files (by mtime) to focus
        diagnostics on the current training window, not the full history.
        """
        trajectories = []
        if not os.path.exists(self.trajectory_dir):
            return trajectories

        # Sort by modification time (newest first), take recent files
        jsonl_files = []
        for fname in os.listdir(self.trajectory_dir):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(self.trajectory_dir, fname)
            jsonl_files.append((os.path.getmtime(fpath), fpath))
        jsonl_files.sort(reverse=True)

        # Load at most the 5 most recent files (roughly last M steps)
        max_files = 5
        for _, fpath in jsonl_files[:max_files]:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trajectories.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return trajectories

    def _compute_training_stats(self, trajectories: List[Dict]) -> Dict:
        """Comprehensive training diagnostics — critical for 8B model debugging.

        Produces:
        - Per-type success rates, avg steps, step distribution
        - Per-app success breakdown (spotify, venmo, multi_app, file_system)
        - Error categorization (auth, API, format, incomplete, loop)
        - Completion behavior (how episodes end)
        - Step distribution for success vs failure
        """
        by_type = defaultdict(list)
        by_app = defaultdict(list)
        error_categories = defaultdict(int)
        completion_modes = defaultdict(int)  # how episodes end
        success_steps = []
        failure_steps = []

        for t in trajectories:
            task_type = t.get("task_type", "unknown")
            success = t.get("success", False)
            n_steps = t.get("n_steps", 0)
            sgc = t.get("evaluation", {}).get("sgc", 0.0)
            tgc = t.get("evaluation", {}).get("tgc", sgc)

            entry = {"success": success, "n_steps": n_steps, "sgc": sgc, "tgc": tgc}
            by_type[task_type].append(entry)

            # Per-app breakdown (from allowed_apps or task_type)
            apps = t.get("allowed_apps", [])
            if not apps:
                apps = [task_type] if task_type != "unknown" else ["unknown"]
            for app in apps:
                by_app[app.lower()].append(entry)

            # Step tracking
            if success:
                success_steps.append(n_steps)
            else:
                failure_steps.append(n_steps)

            # Error categorization (from trajectory steps)
            trajectory_steps = t.get("trajectory", [])
            if not success and trajectory_steps:
                error_type = self._categorize_failure(trajectory_steps, n_steps, t)
                error_categories[error_type] += 1

            # Completion mode
            if success:
                completion_modes["success_complete_task"] += 1
            elif n_steps >= 30:
                completion_modes["timeout_max_steps"] += 1
            elif trajectory_steps:
                last_action = trajectory_steps[-1].get("action", trajectory_steps[-1].get("code", ""))
                if "complete_task" in last_action:
                    completion_modes["wrong_answer"] += 1
                elif "status=\"fail\"" in last_action or "status='fail'" in last_action:
                    completion_modes["gave_up"] += 1
                else:
                    completion_modes["incomplete"] += 1
            else:
                completion_modes["no_trajectory"] += 1

        stats = {
            "overall_success_rate": float(np.mean([t.get("success", False) for t in trajectories])),
            "overall_avg_steps": float(np.mean([t.get("n_steps", 0) for t in trajectories])),
            "n_trajectories": len(trajectories),
            "per_type": {},
            "per_app": {},
            "error_categories": dict(error_categories),
            "completion_modes": dict(completion_modes),
            "step_distribution": {
                "success_avg": float(np.mean(success_steps)) if success_steps else 0.0,
                "success_median": float(np.median(success_steps)) if success_steps else 0.0,
                "failure_avg": float(np.mean(failure_steps)) if failure_steps else 0.0,
                "failure_median": float(np.median(failure_steps)) if failure_steps else 0.0,
                "success_count": len(success_steps),
                "failure_count": len(failure_steps),
            },
        }

        for task_type, entries in by_type.items():
            sr = np.mean([e["success"] for e in entries])
            avg_steps = np.mean([e["n_steps"] for e in entries])
            stats["per_type"][task_type] = {
                "success_rate": float(sr),
                "avg_steps": float(avg_steps),
                "count": len(entries),
                "avg_sgc": float(np.mean([e["sgc"] for e in entries])),
                "avg_tgc": float(np.mean([e["tgc"] for e in entries])),
            }

        for app, entries in by_app.items():
            sr = np.mean([e["success"] for e in entries])
            stats["per_app"][app] = {
                "success_rate": float(sr),
                "count": len(entries),
            }

        # Weakest areas summary (for evolution prompt)
        if stats["per_type"]:
            sorted_types = sorted(stats["per_type"].items(), key=lambda x: x[1]["success_rate"])
            stats["weakest_types"] = [
                {"type": t, "success_rate": v["success_rate"], "count": v["count"]}
                for t, v in sorted_types[:3]
            ]
            stats["weakest_type_success_rate"] = sorted_types[0][1]["success_rate"]

        return stats

    @staticmethod
    def _categorize_failure(trajectory_steps: List[Dict], n_steps: int, traj: Dict) -> str:
        """Categorize failure mode from trajectory steps.

        Categories:
        - auth_error: login/token issues
        - api_error: wrong API calls, bad parameters
        - format_error: code extraction failed, invalid code
        - loop: repeated same action
        - incomplete: ran out of steps without completing
        - wrong_answer: completed but wrong result
        """
        actions = []
        observations = []
        for step in trajectory_steps:
            act = step.get("action", step.get("code", ""))
            obs = step.get("observation", step.get("output", ""))
            actions.append(act)
            observations.append(obs.lower() if isinstance(obs, str) else "")

        all_obs = " ".join(observations)

        # Auth errors
        if any(kw in all_obs for kw in ["unauthorized", "invalid", "login", "access_token",
                                         "not logged in", "authentication"]):
            return "auth_error"

        # Loop detection (>= 3 consecutive identical actions)
        if len(actions) >= 3:
            for i in range(len(actions) - 2):
                if actions[i] == actions[i+1] == actions[i+2] and actions[i]:
                    return "loop"

        # API errors
        if any(kw in all_obs for kw in ["error", "exception", "traceback", "typeerror",
                                         "keyerror", "attributeerror", "nameerror"]):
            return "api_error"

        # Format errors (no code extracted)
        if any(not act.strip() for act in actions):
            return "format_error"

        # Timeout
        if n_steps >= 30:
            return "incomplete"

        return "other"

    def _compute_sphere_diagnostics(self) -> Dict:
        """Sphere spatial info: coverage gaps, redundancy, drift patterns."""
        info = {}
        try:
            summary = self.sphere.summary()
            info["n_skills"] = summary.get("n_skills", 0)
            info["coverage"] = summary.get("coverage", 0.0)

            # Redundancy: pairs with cosine > 0.9 (using sphere vectors)
            redundant = []
            skills = self.sphere.skills if hasattr(self.sphere, "skills") else []
            vectors = self.sphere.vectors if hasattr(self.sphere, "vectors") else None
            if vectors is not None and len(skills) > 0:
                import torch
                sims = torch.mm(vectors, vectors.T)
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        cos = float(sims[i, j])
                        if cos > 0.9:
                            redundant.append({
                                "skill_a": skills[i].name,
                                "skill_b": skills[j].name,
                                "cosine": round(cos, 3),
                            })
            info["redundant_pairs"] = redundant
            info["n_redundant"] = len(redundant)

            # Coverage gap: skills with high isolation (far from neighbors)
            if vectors is not None and len(skills) > 1:
                # For each skill, find distance to nearest neighbor
                sims_no_diag = sims.clone()
                sims_no_diag.fill_diagonal_(-1.0)
                max_sims, _ = sims_no_diag.max(dim=1)
                # Skills with low max-sim to any other skill = isolated
                isolated = []
                for i, ms in enumerate(max_sims):
                    if float(ms) < 0.5:  # cosine < 0.5 = significant gap
                        isolated.append({
                            "skill": skills[i].name,
                            "nearest_sim": round(float(ms), 3),
                        })
                info["isolated_skills"] = isolated

        except Exception as e:
            info["error"] = str(e)

        return info

    def _compute_skill_scores(self, trajectories: List[Dict]) -> Dict:
        """Per-skill usage and effectiveness from trajectory data.

        Tracks: how often a skill is injected, and whether the episode
        it appeared in was successful (outcome-level signal).
        """
        # Track per-skill: [episode_outcome, ...]
        skill_outcomes = defaultdict(list)
        for t in trajectories:
            episode_success = 1.0 if t.get("success", False) else 0.0
            seen_skills = set()
            for step_data in t.get("trajectory", []):
                signals = step_data.get("sphere_signals", {})
                for skill_name in signals.get("skills_used", []):
                    seen_skills.add(skill_name)
            # Each skill used in this episode gets the episode outcome
            for skill_name in seen_skills:
                skill_outcomes[skill_name].append(episode_success)

        result = {}
        for skill, outcomes in skill_outcomes.items():
            result[skill] = {
                "count": len(outcomes),
                "success_rate": float(np.mean(outcomes)),
                "episodes_used": len(outcomes),
            }
        return result

    # ── Step 2: Generate Candidates ──────────────────────────────────────

    def generate_candidates(
        self,
        diagnostics: Dict,
        step: int,
        update_training: bool = False,
    ) -> List[Dict]:
        """Generate G candidate skill sets.

        Phase 1: All candidates from API (GPT-5.4).
        Phase 2: Mixed — n_model from local vLLM, rest from API.

        Each candidate = {behavioral_changes: [...], training_skill: {...}}
        If update_training=False, training skill is locked to current active.
        """
        n_model = int(self.G * self.model_candidate_ratio)
        n_api = self.G - n_model

        candidates = []

        # Model-generated candidates (Phase 2)
        if n_model > 0:
            model_cands = self._generate_model_candidates(
                diagnostics, n_model, update_training)
            candidates.extend(model_cands)

        # API-generated candidates (Phase 1, or remaining in Phase 2)
        if n_api > 0:
            api_cands = self._generate_api_candidates(
                diagnostics, n_api, update_training)
            candidates.extend(api_cands)

        print(f"[ATS Outer] Generated {len(candidates)}/{self.G} candidates "
              f"(model={n_model}, api={n_api}, training_update={update_training})")
        return candidates

    def _generate_api_candidates(
        self, diagnostics: Dict, n: int, update_training: bool,
    ) -> List[Dict]:
        """Generate candidates via API (GPT-5.4)."""
        prompt = self._build_evolution_prompt(diagnostics, update_training)
        candidates = []
        for g in range(n):
            try:
                response = self.client.chat.completions.create(
                    model=self.verifier_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4000,
                    temperature=0.7,
                )
                text = response.choices[0].message.content.strip()
                candidate = self._parse_candidate(text, update_training)
                candidate["candidate_id"] = g
                candidate["source"] = "api"
                candidates.append(candidate)
            except Exception as e:
                print(f"[ATS Outer] API candidate {g} failed: {e}")
        return candidates

    def _generate_model_candidates(
        self, diagnostics: Dict, n: int, update_training: bool,
    ) -> List[Dict]:
        """Generate candidates via local vLLM model (Phase 2).

        Requires vLLM server running at localhost:8000.
        """
        try:
            from skill_sphere.agent.llm_client import LLMClient
        except ImportError:
            print("[ATS Outer] LLMClient not available, falling back to API")
            return self._generate_api_candidates(diagnostics, n, update_training)

        # Auto-detect model name from vLLM
        model_name = self.proxy_model_path or "Qwen/Qwen3-8B"
        try:
            import urllib.request
            resp = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
            models_data = json.loads(resp.read())
            if models_data.get("data"):
                model_name = models_data["data"][0]["id"]
        except Exception:
            pass

        llm = LLMClient(
            base_url="http://localhost:8000/v1",
            model=model_name,
        )

        prompt = self._build_evolution_prompt(diagnostics, update_training)
        candidates = []
        for g in range(n):
            try:
                temp = 0.7 + g * 0.1
                response = llm.generate(
                    system_prompt="You are a skill evolution expert. Output valid JSON only.",
                    user_prompt=prompt,
                    temperature=temp,
                )
                candidate = self._parse_candidate(response, update_training)
                candidate["candidate_id"] = self.G - n + g  # offset by API candidates
                candidate["source"] = "model"
                candidates.append(candidate)
            except Exception as e:
                print(f"[ATS Outer] Model candidate {g} failed: {e}")
        return candidates

    def _build_evolution_prompt(self, diagnostics: Dict, update_training: bool) -> str:
        """Build prompt for API to generate skill evolution candidates."""
        stats = diagnostics.get("training_stats", {})
        sphere_info = diagnostics.get("sphere_info", {})
        skill_scores = diagnostics.get("skill_scores", {})

        current_skills_text = json.dumps(
            [{"title": s.get("title"), "guidance": s.get("principle", s.get("guidance", ""))[:200]}
             for s in self.behavioral_skills],
            indent=2, ensure_ascii=False
        )

        training_skill_text = json.dumps(self.active_training_skill, indent=2, ensure_ascii=False) \
            if self.active_training_skill else "None"

        # Build concise weakest-areas summary for the prompt
        weakest = stats.get("weakest_types", [])
        weakest_text = "\n".join(
            f"  - {w['type']}: {w['success_rate']:.1%} ({w['count']} episodes)"
            for w in weakest
        ) if weakest else "  (no data)"

        error_cats = stats.get("error_categories", {})
        error_text = ", ".join(f"{k}: {v}" for k, v in sorted(error_cats.items(), key=lambda x: -x[1])) \
            if error_cats else "(no failures analyzed)"

        completion = stats.get("completion_modes", {})
        completion_text = ", ".join(f"{k}: {v}" for k, v in sorted(completion.items(), key=lambda x: -x[1])) \
            if completion else "(no data)"

        step_dist = stats.get("step_distribution", {})

        # Pre-compute conditional strings to avoid backslash-in-fstring issues (Python 3.10)
        training_skill_json_line = (
            '"training_skill": {"title": "...", "data_selection": "...", "reward_formula": "...", "when_to_use": "..."},'
            if update_training else ""
        )
        training_skill_rule_line = (
            "- You may also propose a new training skill or modify the active one"
            if update_training else "- Training skill is LOCKED, do not modify it"
        )

        prompt = f"""You are an AI training skill evolver. Analyze the diagnostics and propose improved skills.

## Current Training Status
- Overall success rate: {stats.get('overall_success_rate', 0):.1%}
- Avg steps: {stats.get('overall_avg_steps', 0):.1f}
- Total trajectories: {stats.get('n_trajectories', 0)}
- Success avg steps: {step_dist.get('success_avg', 0):.1f} (median {step_dist.get('success_median', 0):.0f})
- Failure avg steps: {step_dist.get('failure_avg', 0):.1f} (median {step_dist.get('failure_median', 0):.0f})

## Weakest Areas (prioritize these)
{weakest_text}

## Error Categories (failure root causes)
{error_text}

## Episode Completion Modes
{completion_text}

## Per-Type Breakdown
{json.dumps(stats.get('per_type', {}), indent=2)}

## Per-App Breakdown
{json.dumps(stats.get('per_app', {}), indent=2)}

## Current Behavioral Skills ({len(self.behavioral_skills)})
{current_skills_text}

## Skill Usage Stats
{json.dumps(dict(skill_scores), indent=2)}

## Sphere Spatial Info
{json.dumps(sphere_info, indent=2)}

## Current Active Training Skill
{training_skill_text}

## Your Task
Propose ONE candidate skill configuration. Output valid JSON with this structure:
{{
  "reasoning": "1-2 sentences on why these changes",
  "behavioral_changes": [
    {{"action": "modify|add|remove", "title": "...", "guidance": "...", "scoring": "...", "when_to_use": "...", "category": "..."}}
  ],
  {training_skill_json_line}
}}

Rules:
- Behavioral skill Guidance must be CONCRETE (specific API calls, code patterns), not abstract principles
- Focus changes on weakest task types (lowest success rate)
- If sphere shows coverage gaps, add skills for uncovered areas
- If sphere shows redundancy (cosine>0.9), merge or remove one
- Maximum 3 changes per candidate (modify/add/remove)
{training_skill_rule_line}
"""
        return prompt

    def _parse_candidate(self, text: str, update_training: bool) -> Dict:
        """Parse API response into structured candidate."""
        # Try to extract JSON from response — find balanced braces
        import re
        # Try json.loads on progressively larger substrings from first {
        start = text.find('{')
        parsed = None
        if start >= 0:
            # Try nested brace matching for robustness
            depth = 0
            for end in range(start, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:end+1])
                            break
                        except json.JSONDecodeError:
                            continue

        if parsed:
            candidate = {
                "reasoning": parsed.get("reasoning", ""),
                "behavioral_changes": parsed.get("behavioral_changes", []),
            }
            if update_training and "training_skill" in parsed:
                candidate["training_skill"] = parsed["training_skill"]
            return candidate

        # Fallback: no changes
        print(f"[ATS Outer] WARNING: Failed to parse candidate JSON from API response")
        return {"reasoning": "Failed to parse", "behavioral_changes": []}

    # ── Step 3: Proxy Evaluation ─────────────────────────────────────────

    def _sample_proxy_tasks(self) -> List[str]:
        """Sample a fixed set of task IDs for proxy evaluation.

        Called once per evolution step so all candidates are compared on the same tasks.
        """
        appworld_root = os.environ.get("APPWORLD_ROOT", "")
        if not appworld_root:
            raise EnvironmentError(
                "APPWORLD_ROOT environment variable must be set "
                "(e.g. export APPWORLD_ROOT=/path/to/srpo)"
            )
        task_file = f"{appworld_root}/data/datasets/train.txt"
        if os.path.exists(task_file):
            with open(task_file) as f:
                all_ids = [l.strip() for l in f if l.strip()]
            rng = np.random.RandomState(int(time.time()) % 2**31)
            rng.shuffle(all_ids)
            return all_ids[:self.proxy_eval_tasks]
        print("[ATS Outer] No train tasks found for proxy eval")
        return []

    def proxy_eval(self, candidate: Dict, task_ids: List[str] = None) -> float:
        """Evaluate candidate by applying its skills and running small task batch.

        Returns average outcome reward as proxy score.
        """
        if not task_ids:
            task_ids = self._sample_proxy_tasks()
        if not task_ids:
            return 0.0

        # Apply candidate's behavioral changes to create modified skill set
        modified_skills = self._apply_candidate(candidate)

        # Write modified skills to temp file for the agent's sphere
        import tempfile
        tmp_skills_file = os.path.join(self.output_dir, "_proxy_eval_skills.json")
        with open(tmp_skills_file, "w") as f:
            json.dump(modified_skills, f, indent=2, ensure_ascii=False)

        # Run evaluation with modified skills
        try:
            from skill_sphere.agent.appworld_agent import AppWorldAgent
            from skill_sphere.agent.llm_client import LLMClient
            from skill_sphere.skill_bank.skill_sphere import SkillSphere
            from skill_sphere.skill_bank.encoder import SkillEncoder
            from skill_sphere.env.appworld_wrapper import AppWorldEnv

            # Model name = path used by vLLM (auto-detect from /v1/models)
            model_name = self.proxy_model_path or "Qwen/Qwen3-8B"
            try:
                import urllib.request, json as _json
                resp = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                models_data = _json.loads(resp.read())
                if models_data.get("data"):
                    model_name = models_data["data"][0]["id"]
            except Exception:
                pass
            llm = LLMClient(
                base_url="http://localhost:8000/v1",
                model=model_name,
            )

            # Build sphere from modified skills
            encoder = SkillEncoder(device="cpu")
            sphere = SkillSphere.from_skillrl_json(
                tmp_skills_file, encoder=encoder, device="cpu"
            )

            successes = 0
            for task_id in task_ids:
                try:
                    agent = AppWorldAgent(
                        llm=llm,
                        skill_sphere=sphere,
                        mode="sphere",
                        max_history=5,
                    )
                    env = AppWorldEnv(
                        experiment_name=f"proxy_eval_{task_id}",
                        max_interactions=15,
                    )
                    result = agent.run_task(env, task_id)
                    if result.get("success", False):
                        successes += 1
                    env.close()
                except Exception as e:
                    print(f"[ATS Outer] Proxy eval error {task_id}: {e}")

            proxy_reward = successes / max(1, len(task_ids))
            print(f"[ATS Outer] Proxy eval: {successes}/{len(task_ids)} = {proxy_reward:.2%}")
            return proxy_reward

        except (ImportError, ConnectionError, OSError) as e:
            print(f"[ATS Outer] Cannot run proxy eval ({type(e).__name__}): {e}")
            # Fallback: score based on change quality heuristics
            return self._heuristic_score(candidate)
        except Exception as e:
            print(f"[ATS Outer] Proxy eval unexpected error: {e}")
            return self._heuristic_score(candidate)

    def _heuristic_score(self, candidate: Dict) -> float:
        """Fallback scoring when env is unavailable."""
        changes = candidate.get("behavioral_changes", [])
        if not changes:
            return 0.3
        # Simple heuristic: more targeted changes score higher
        score = 0.5
        for c in changes:
            if c.get("action") == "modify":
                score += 0.1
            elif c.get("action") == "add":
                score += 0.05
        return min(1.0, score)

    def _apply_candidate(self, candidate: Dict) -> List[Dict]:
        """Apply candidate's behavioral changes to current skill set."""
        import copy
        skills = copy.deepcopy(self.behavioral_skills)

        for change in candidate.get("behavioral_changes", []):
            action = change.get("action", "")
            title = change.get("title", "")

            if action == "modify":
                for s in skills:
                    if s.get("title") == title:
                        if "guidance" in change:
                            s["principle"] = change["guidance"]
                            s["guidance"] = change["guidance"]
                        if "scoring" in change:
                            s["scoring"] = change["scoring"]
                        if "when_to_use" in change:
                            s["when_to_use"] = change["when_to_use"]
                        break

            elif action == "add":
                new_skill = {
                    "title": title,
                    "type": "behavioral",
                    "principle": change.get("guidance", ""),
                    "guidance": change.get("guidance", ""),
                    "scoring": change.get("scoring", ""),
                    "when_to_use": change.get("when_to_use", ""),
                    "category": change.get("category", "appworld/general"),
                }
                # Validate: new skills must be scorable (title + scoring >= 20 chars)
                if not title.strip():
                    print(f"[ATS Outer] Skipping add: empty title")
                    continue
                scoring_text = new_skill.get("scoring", "")
                if len(scoring_text) < 20:
                    print(f"[ATS Outer] Skipping add '{title}': scoring too short ({len(scoring_text)} chars)")
                    continue
                skills.append(new_skill)

            elif action == "remove":
                skills = [s for s in skills if s.get("title") != title]

        return skills

    # ── Step 4: Select Best + Update ─────────────────────────────────────

    def select_and_update(
        self,
        candidates: List[Dict],
        proxy_scores: List[float],
        step: int,
        update_training: bool = False,
    ) -> Dict:
        """Select best candidate and update active skills."""
        if not candidates:
            return {"action": "no_candidates"}

        best_idx = int(np.argmax(proxy_scores))
        best = candidates[best_idx]
        best_score = proxy_scores[best_idx]

        print(f"[ATS Outer] Best candidate {best_idx}: score={best_score:.3f}")
        print(f"[ATS Outer] Reasoning: {best.get('reasoning', '')}")

        # Skip update if best candidate scores 0 (no improvement possible)
        if best_score <= 0.0:
            print("[ATS Outer] Best candidate scored 0.0, skipping update")
            return {"action": "skipped_low_score", "step": step, "best_score": best_score}

        # Apply behavioral changes
        self.behavioral_skills = self._apply_candidate(best)

        # Update training skill if allowed
        if update_training and "training_skill" in best:
            ts = best["training_skill"]
            new_skill_id = f"train_evolved_step{step}"
            self.active_training_skill = {
                "skill_id": new_skill_id,
                "title": ts.get("title", "evolved"),
                "type": "training",
                "data_selection": ts.get("data_selection", "balanced"),
                "reward_formula": ts.get("reward_formula", ""),
                "reward_weights": ts.get("reward_weights", {}),
                "when_to_use": ts.get("when_to_use", ""),
                "active": True,
            }
            # Deactivate others
            for s in self.training_skills:
                s["active"] = False
            self.training_skills.append(self.active_training_skill)
            self._active_skill_id = new_skill_id

        # Save updated skills
        self._save_skills(step)

        # Re-encode embeddings for sphere
        self._reindex_sphere()

        # Record history
        record = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candidates": len(candidates),
            "best_idx": best_idx,
            "best_score": best_score,
            "all_scores": proxy_scores,
            "reasoning": best.get("reasoning", ""),
            "changes": best.get("behavioral_changes", []),
            "update_training": update_training,
        }
        self.history.append(record)

        # Save history
        history_path = os.path.join(self.output_dir, "evolution_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        return record

    def _save_skills(self, step: int):
        """Save updated behavioral and training skills."""
        # Save current version
        with open(self.skills_path, "w") as f:
            json.dump(self.behavioral_skills, f, indent=2, ensure_ascii=False)

        # Preserve dict format with active_skill_id
        active_id = (self.active_training_skill.get("skill_id", "")
                     if self.active_training_skill else "")
        with open(self.training_skills_path, "w") as f:
            json.dump({
                "training_skills": self.training_skills,
                "active_skill_id": active_id,
            }, f, indent=2, ensure_ascii=False)

        # Save versioned copy
        versioned = os.path.join(self.output_dir, f"skills_step{step}.json")
        with open(versioned, "w") as f:
            json.dump({
                "behavioral": self.behavioral_skills,
                "training": self.training_skills,
                "step": step,
            }, f, indent=2, ensure_ascii=False)

        print(f"[ATS Outer] Saved {len(self.behavioral_skills)} behavioral + "
              f"{len(self.training_skills)} training skills")

    def _reindex_sphere(self):
        """Re-encode modified skill embeddings into sphere."""
        if self.sphere is None:
            return
        try:
            # Re-build sphere from updated skills
            from skill_sphere.skill_bank.skill_sphere import SkillSphere
            self.sphere = SkillSphere.from_skillrl_json(self.skills_path)
            print(f"[ATS Outer] Sphere re-indexed: {self.sphere.summary()}")
        except Exception as e:
            print(f"[ATS Outer] Sphere re-index failed: {e}")

    # ── Proxy Eval Only (for Outer GRPO) ─────────────────────────────────

    def proxy_eval_candidates(
        self, candidates_json_path: str, step: int
    ) -> Dict:
        """Proxy eval pre-generated candidates (from OuterGRPO).

        Loads candidates from JSON, runs proxy eval, selects best,
        updates skill bank, and returns results with per-candidate rewards.
        """
        print(f"[ATS Outer] Proxy eval mode: loading {candidates_json_path}")
        with open(candidates_json_path) as f:
            candidates = json.load(f)

        if not candidates:
            return {"action": "no_candidates", "proxy_scores": []}

        # Should we update training skill?
        update_training = (step % (self.K * self.M) == 0) and (step > 0)

        # Parse candidate text into structured format
        parsed_candidates = []
        for c in candidates:
            text = c.get("text", "")
            try:
                parsed = self._parse_candidate(text, update_training)
                parsed["candidate_id"] = c.get("candidate_id", len(parsed_candidates))
                parsed["source"] = c.get("source", "model_grpo")
                parsed_candidates.append(parsed)
            except Exception as e:
                print(f"[ATS Outer] Failed to parse candidate {c.get('candidate_id', '?')}: {e}")
                parsed_candidates.append({
                    "candidate_id": c.get("candidate_id", len(parsed_candidates)),
                    "source": "model_grpo",
                    "reasoning": "",
                    "behavioral_changes": [],
                })

        # Proxy eval all candidates on same task set
        proxy_task_ids = self._sample_proxy_tasks()
        proxy_scores = []

        if not proxy_task_ids:
            proxy_scores = [0.5] * len(parsed_candidates)
            print("[ATS Outer] No proxy eval tasks available, using default scores")
        else:
            for c in parsed_candidates:
                score = self.proxy_eval(c, task_ids=proxy_task_ids)
                proxy_scores.append(score)

        # Select best and update skill bank
        result = self.select_and_update(
            parsed_candidates, proxy_scores, step, update_training
        )
        result["proxy_scores"] = proxy_scores

        # Save proxy scores for outer GRPO to read
        scores_path = os.path.join(self.output_dir, f"proxy_scores_step{step}.json")
        with open(scores_path, "w") as f:
            json.dump({
                "proxy_scores": proxy_scores,
                "best_idx": int(np.argmax(proxy_scores)) if proxy_scores else 0,
            }, f, indent=2)
        print(f"[ATS Outer] Proxy scores saved: {scores_path}")

        return result

    # ── Main Entry: Run One Evolution Step ───────────────────────────────

    def run_step(self, step: int) -> Dict:
        """Run one full outer loop evolution step.

        Called every M inner steps. Checks if training skill should update too.
        """
        print(f"\n{'='*60}")
        print(f"[ATS Outer] Evolution step at inner step {step}")
        print(f"{'='*60}")

        # Should we update training skill?
        update_training = (step % (self.K * self.M) == 0) and (step > 0)

        # 1. Diagnostics
        diagnostics = self.collect_diagnostics(step)
        diag_path = os.path.join(self.output_dir, f"diagnostics_step{step}.json")
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)

        # 2. Generate candidates
        candidates = self.generate_candidates(diagnostics, step, update_training)
        if not candidates:
            print("[ATS Outer] No valid candidates, skipping")
            return {"action": "skipped", "step": step}

        # 3. Proxy eval (same task set for all candidates for fair comparison)
        proxy_task_ids = self._sample_proxy_tasks()
        if not proxy_task_ids:
            # No proxy eval: randomly pick one candidate (all equally valid without eval)
            import random
            winner = random.randrange(len(candidates))
            proxy_scores = [0.5] * len(candidates)
            proxy_scores[winner] = 1.0
            print(f"[ATS Outer] No proxy eval tasks, random winner: candidate {winner}")
        else:
            proxy_scores = []
            for c in candidates:
                score = self.proxy_eval(c, task_ids=proxy_task_ids)
                proxy_scores.append(score)

        # 4. Select best + update
        result = self.select_and_update(candidates, proxy_scores, step, update_training)

        return result

    def get_active_config(self) -> Dict:
        """Get current active configuration for inner loop.

        Returns reward formula weights + data selection policy.
        """
        config = {
            "data_selection": "balanced",
            "reward_formula": {
                "outcome": 1.0,
                "supervision": 0.3,
                "efficiency": 0.0,
            },
        }

        if self.active_training_skill:
            ts = self.active_training_skill
            config["data_selection"] = ts.get("data_selection", "balanced")

            # Prefer structured reward_weights dict; fallback to parsing formula text
            weights = ts.get("reward_weights")
            if isinstance(weights, dict):
                for component in ["outcome", "supervision", "efficiency"]:
                    if component in weights:
                        config["reward_formula"][component] = float(weights[component])
            else:
                formula_text = ts.get("reward_formula", "")
                import re
                for component in ["outcome", "supervision", "efficiency"]:
                    match = re.search(rf"([\d.]+)\s*[×*]\s*{component}", formula_text)
                    if match:
                        config["reward_formula"][component] = float(match.group(1))

        return config


def main():
    parser = argparse.ArgumentParser(description="ATS Outer Evolution Loop")
    parser.add_argument("--skills_path", default="data/skills/appworld_skills_ats.json")
    parser.add_argument("--training_skills_path", default="data/skills/training_skills.json")
    parser.add_argument("--trajectory_dir", default="results/ats_grpo")
    parser.add_argument("--step", type=int, required=True, help="Current inner loop step")
    parser.add_argument("--M", type=int, default=10, help="Behavioral update frequency")
    parser.add_argument("--K", type=int, default=3, help="Training update multiplier")
    parser.add_argument("--G", type=int, default=3, help="Number of candidates")
    parser.add_argument("--proxy_eval_tasks", type=int, default=10)
    parser.add_argument("--sphere_path", default=None)
    parser.add_argument("--output_dir", default="results/ats_evolution")
    parser.add_argument("--verifier_model", default="gpt-5.4")
    parser.add_argument("--proxy_model_path", default=None,
                        help="Model path for proxy eval (auto-detect from vLLM if None)")
    parser.add_argument("--model_candidate_ratio", type=float, default=0.0,
                        help="Fraction of candidates from model (0=Phase1 all API, 1=Phase2 all model)")
    parser.add_argument("--candidates_json", type=str, default=None,
                        help="Path to pre-generated candidates JSON (proxy eval only mode)")
    parser.add_argument("--proxy_eval_only", action="store_true",
                        help="Only run proxy eval on pre-generated candidates (for outer GRPO)")
    args = parser.parse_args()

    loop = ATSOuterLoop(
        skills_path=args.skills_path,
        training_skills_path=args.training_skills_path,
        trajectory_dir=args.trajectory_dir,
        M=args.M,
        K=args.K,
        G=args.G,
        proxy_eval_tasks=args.proxy_eval_tasks,
        sphere_path=args.sphere_path,
        output_dir=args.output_dir,
        verifier_model=args.verifier_model,
        proxy_model_path=args.proxy_model_path,
        model_candidate_ratio=args.model_candidate_ratio,
    )

    if args.proxy_eval_only and args.candidates_json:
        result = loop.proxy_eval_candidates(args.candidates_json, args.step)
    else:
        result = loop.run_step(args.step)
    print(f"\n[ATS Outer] Result: {json.dumps(result, indent=2)}")

    # Print active config for inner loop
    config = loop.get_active_config()
    print(f"\n[ATS Outer] Active config for inner loop:")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
