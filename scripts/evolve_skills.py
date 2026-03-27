"""
ATS Outer Loop: Skill Evolution

Phase 1 (API-driven): GPT-5.4 reads diagnostic report + current skill library,
generates G candidate skill modification proposals.
Each candidate is evaluated by proxy reward.

Phase 2 (model-driven): Replace GPT-5.4 with the trained model itself.
"""

import argparse
import json
import os
import sys
import copy
from pathlib import Path

from openai import AzureOpenAI

AZURE_CONFIG = {
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    "api_version": "2025-01-01-preview",
}
MODEL = "gpt-5.4"


def load_skills(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_diagnostics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_training_skills(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_skill_library(skills: dict) -> str:
    """Format current skill library as readable text for the API."""
    lines = []
    lines.append("## Current Behavioral Skills\n")

    def _format_skill(s, default_id="?"):
        """Format one skill entry with scoring_type info."""
        sid = s.get('skill_id', default_id)
        stype = s.get('scoring_type', 'llm')
        out = [f"### [{sid}] {s.get('title', '?')} [scoring_type={stype}]"]
        out.append(f"- Principle: {s.get('principle', '')}")
        if s.get('when_to_apply'):
            out.append(f"- When to apply: {s['when_to_apply']}")
        if s.get('scoring'):
            out.append(f"- Scoring: {s['scoring']}")
        if s.get('scoring_rule'):
            out.append(f"- Scoring rule: {json.dumps(s['scoring_rule'])}")
        return out

    for i, s in enumerate(skills.get("general_skills", [])):
        lines.extend(_format_skill(s, f"gen_{i}"))
        lines.append(f"- Category: {s.get('category', 'general')}")
        lines.append("")

    for cat, cat_skills in skills.get("task_specific_skills", {}).items():
        for i, s in enumerate(cat_skills):
            lines.extend(_format_skill(s, f"{cat}_{i}"))
            lines.append(f"- Category: appworld/{cat}")
            lines.append("")

    for i, s in enumerate(skills.get("common_mistakes", [])):
        lines.extend(_format_skill(s, f"cm_{i+1:03d}"))
        lines.append(f"- Category: common_mistakes")
        lines.append("")

    return "\n".join(lines)


def format_diagnostics(report: dict) -> str:
    """Format diagnostic report as readable text."""
    lines = []

    # Overall stats
    overall = report["training_statistics"]["overall"]
    lines.append("## Training Statistics\n")
    lines.append(f"- Total episodes: {overall['total_episodes']}")
    lines.append(f"- Success rate: {overall['success_rate']:.1%}")
    lines.append(f"- Avg steps: {overall['avg_steps']:.1f}")
    lines.append(f"- Avg SGC: {overall['avg_sgc']:.1%}")
    lines.append("")

    # Per task type
    lines.append("### Per Task Type")
    for t, info in report["training_statistics"]["per_task_type"].items():
        lines.append(f"- {t}: {info['success']}/{info['total']} ({info['success_rate']:.0%}), "
                     f"avg {info['avg_steps']:.1f} steps, SGC={info['avg_sgc']:.1%}")
    lines.append("")

    # Per skill scores (if available)
    skill_scores = report["training_statistics"].get("per_skill_scores", {})
    if skill_scores:
        lines.append("### Per Skill Verifier Scores")
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1]["avg_score"])
        for name, info in sorted_skills[:10]:
            lines.append(f"- **LOW** {name}: avg={info['avg_score']:.2f}, std={info['std']:.2f}, n={info['n']}")
        lines.append("...")
        for name, info in sorted_skills[-5:]:
            sat = " [SATURATED]" if info.get("saturated") else ""
            lines.append(f"- **HIGH** {name}: avg={info['avg_score']:.2f}, std={info['std']:.2f}, n={info['n']}{sat}")
        lines.append("")

    # Failure patterns
    fp = report["training_statistics"].get("failure_patterns", {})
    if fp:
        lines.append("### Failure Patterns")
        for pattern, count in sorted(fp.items(), key=lambda x: -x[1]):
            lines.append(f"- {pattern}: {count} occurrences")
        lines.append("")

    # Sphere spatial info
    sphere = report.get("sphere_spatial_info", {})
    if sphere:
        lines.append("## Sphere Spatial Info\n")
        lines.append(f"- Total skills: {sphere.get('n_skills', '?')}")

        # Redundancy
        pairs = sphere.get("redundant_pairs", [])
        if pairs:
            lines.append(f"\n### Redundant Pairs ({len(pairs)} found, cosine > 0.9)")
            for p in pairs:
                lines.append(f"- '{p['skill_a']}' ↔ '{p['skill_b']}' (cos={p['cosine']})")

        # Sparse skills
        sparse = sphere.get("sparse_skills", [])
        if sparse:
            lines.append(f"\n### Coverage Gaps ({len(sparse)} sparse skills)")
            for s in sparse:
                lines.append(f"- '{s['skill']}' (category: {s['category']}, knn_dist={s['avg_knn_distance']})")

        # Drift
        drift = sphere.get("drift_stats", {})
        if drift:
            lines.append(f"\n### Agent Drift")
            lines.append(f"- Avg drift rate: {drift.get('avg_drift_rate', 0):.3f}")
            lines.append(f"- High drift steps: {drift.get('high_drift_steps', 0)}/{drift.get('total_steps_with_drift', 0)}")

    # Skill-success correlation
    corr = report["training_statistics"].get("skill_success_correlation", {})
    if corr:
        lines.append("### Skill-Success Correlation (injected vs not)")
        # Show most helpful and most harmful
        sorted_corr = sorted(
            [(k, v) for k, v in corr.items() if v.get("delta") is not None],
            key=lambda x: x[1]["delta"]
        )
        if sorted_corr:
            for name, info in sorted_corr[:5]:  # most harmful
                lines.append(f"- **HARMFUL** {name}: injected={info['injected_success_rate']:.0%} (n={info['injected_n']}), "
                             f"not={info['not_injected_success_rate']:.0%} (n={info['not_injected_n']}), Δ={info['delta']:+.1%}")
            lines.append("...")
            for name, info in sorted_corr[-3:]:  # most helpful
                lines.append(f"- **HELPFUL** {name}: injected={info['injected_success_rate']:.0%} (n={info['injected_n']}), "
                             f"not={info['not_injected_success_rate']:.0%} (n={info['not_injected_n']}), Δ={info['delta']:+.1%}")
        lines.append("")

    # Skill-intent mismatch
    mismatches = report.get("skill_intent_mismatch", [])
    if mismatches:
        lines.append(f"### Skill-Intent Mismatch ({len(mismatches)} found)")
        for m in mismatches:
            lines.append(f"- '{m['skill']}': injected {m['injection_count']}× but avg_score={m['avg_verifier_score']:.2f}")
        lines.append("")

    # Score trends
    trends = report.get("score_trends", {})
    if trends:
        lines.append("### Score Trends (vs previous outer step)")
        for name, t in sorted(trends.items(), key=lambda x: x[1].get("current", 0)):
            arrow = {"improving": "↑", "declining": "↓", "stable": "→"}.get(t["direction"], "?")
            lines.append(f"- {arrow} {name}: {t['previous']:.2f} → {t['current']:.2f} ({t['direction']})")
        lines.append("")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        lines.append(f"\n## Auto-Generated Recommendations ({len(recs)})")
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines)


EVOLUTION_SYSTEM_PROMPT = """You are an ATS (Adaptive Training Skills) Skill Evolver for an AI agent training system.

Your job: analyze the diagnostic report and current skill library, then propose modifications to improve BOTH behavioral skills AND training skills.

The agent operates in AppWorld (multi-app API interaction tasks). Skills guide the agent's behavior at each step via sphere-based retrieval injection.

## Two Skill Types

### Behavioral Skills (per-step, sphere-injected)
Each has:
- **title**: concise name
- **principle**: detailed behavioral guidance (must be specific to API-level, not abstract)
- **when_to_apply**: when this skill is relevant
- **scoring**: evaluation criteria for verifier (score 0-1)
- **scoring_type**: "programmatic" or "llm" (see below)
- **scoring_rule**: (only for programmatic) structured rule for code-based scoring
- **category**: general | appworld/spotify | appworld/venmo | appworld/multi_app | appworld/file_system | common_mistakes

### Scoring Type Selection
Choose scoring_type based on what the scoring criteria requires:
- **"programmatic"**: Use when scoring can be done by counting/detecting patterns in the trajectory text (e.g., "count API doc calls", "check if credential API was called before login"). MUST also provide scoring_rule.
- **"llm"**: Use when scoring requires qualitative judgment (e.g., "did the agent plan effectively", "was error handling graceful"). No scoring_rule needed.

Available scoring_rule types:
- `count_ratio`: Count pattern occurrences, normalize by apps. Fields: count_patterns (list[str]), normalize_by ("unique_apps"), thresholds ({perfect: N, partial: N})
- `pattern_before`: Check if required pattern appears before target. Fields: required_before (list[str]), target_action (str), score_if_present (float), score_if_absent (float)
- `count_penalty`: Penalize for pattern occurrences. Fields: penalty_patterns (list[str]), thresholds ({zero: N, one: N})
- `order_check`: Check if list patterns appear before detail patterns. Fields: list_patterns (list[str]), detail_patterns (list[str])

### Training Skills (global, 1 active at a time)
Each has:
- **skill_id**: unique id (train_XXX)
- **title**: strategy name
- **data_selection**: which tasks to sample for training (natural language, program-parseable)
- **reward_formula**: how to combine outcome + supervision + efficiency into reward (math expression)
- **reward_weights**: parsed weights dict, e.g. {"outcome": 1.0, "supervision": 0.3}
- **when_to_use**: condition for activating this training skill
- **when_to_use_condition**: structured condition for auto-switching

## What You Can Do

### For Behavioral Skills:
1. **MODIFY**: Change an existing skill's principle/scoring/when_to_apply (give skill_id + new content)
2. **ADD**: Create a new skill (give full skill definition)
3. **REMOVE**: Archive a skill that is saturated or redundant (give skill_id)
4. **MERGE**: Combine two redundant skills into one (give both skill_ids + merged skill)

### For Training Skills (only when training_skill_locked=false):
5. **MODIFY_TRAINING**: Change a training skill's data_selection/reward_formula/reward_weights
6. **ADD_TRAINING**: Create a new training skill
7. **SWITCH_TRAINING**: Change which training skill is active (give skill_id)

## Constraints
- Keep behavioral skills between 30-60
- Training skills should stay between 2-6 (small library, 1 active)
- Prioritize fixing LOW-scoring skills, addressing failure patterns, and harmful skills (negative Δ in correlation)
- Skills marked MISMATCH need content rewriting (position is right, content is wrong)
- New skills must have specific, actionable Guidance (not abstract principles)
- Scoring criteria must be verifiable from trajectory data
- For new/modified skills: if scoring involves counting or pattern detection, set scoring_type="programmatic" and provide scoring_rule; otherwise set scoring_type="llm"
- Consider sphere spatial info: merge redundant pairs, fill coverage gaps
- Training skill modifications: adjust reward weights based on current training stage, refine data selection based on per-type success rates

## Output Format
Return a JSON object with:
```json
{
  "reasoning": "Brief analysis of what needs to change and why",
  "behavioral_modifications": [
    {"action": "modify", "skill_id": "gen_001", "changes": {"principle": "...", "scoring": "..."}},
    {"action": "add", "skill": {"title": "...", "principle": "...", "when_to_apply": "...", "scoring": "...", "scoring_type": "llm_or_programmatic", "scoring_rule": {"type": "...", ...}, "category": "..."}},
    {"action": "remove", "skill_id": "cm_003", "reason": "..."},
    {"action": "merge", "skill_ids": ["gen_003", "cm_005"], "merged_skill": {"title": "...", ...}}
  ],
  "training_modifications": [
    {"action": "modify_training", "skill_id": "train_001", "changes": {"reward_formula": "...", "reward_weights": {...}, "data_selection": "..."}},
    {"action": "add_training", "skill": {"skill_id": "train_004", "title": "...", "data_selection": "...", "reward_formula": "...", "reward_weights": {...}, "when_to_use": "...", "when_to_use_condition": {...}}},
    {"action": "switch_training", "skill_id": "train_002"}
  ]
}
```
Note: If training_skill_locked=true, return empty training_modifications list.
"""


def generate_candidates(
    client: AzureOpenAI,
    skills: dict,
    diagnostics: dict,
    training_skills: dict,
    n_candidates: int = 3,
    training_skill_locked: bool = False,
) -> list[dict]:
    """Generate G candidate skill modification proposals (behavioral + training)."""

    skill_text = format_skill_library(skills)
    diag_text = format_diagnostics(diagnostics)

    # Full training skill library context
    training_context = "\n## Training Skills Library\n"
    active_id = training_skills.get("active_skill_id", "train_001")
    for ts in training_skills.get("training_skills", []):
        is_active = " **[ACTIVE]**" if ts["skill_id"] == active_id else ""
        training_context += (
            f"\n### [{ts['skill_id']}] {ts['title']}{is_active}\n"
            f"- Data Selection: {ts['data_selection']}\n"
            f"- Reward Formula: {ts['reward_formula']}\n"
            f"- Reward Weights: {json.dumps(ts.get('reward_weights', {}))}\n"
            f"- When to Use: {ts['when_to_use']}\n"
        )

    lock_status = (
        f"\n**training_skill_locked = {'true' if training_skill_locked else 'false'}**\n"
        f"{'Training skills cannot be modified this round (behavioral only).' if training_skill_locked else 'Training skills CAN be modified/added/switched this round.'}\n"
    )

    user_prompt = (
        f"# Current Behavioral Skill Library\n\n{skill_text}\n\n"
        f"# Current Training Skills\n\n{training_context}\n{lock_status}\n"
        f"# Diagnostic Report\n\n{diag_text}\n\n"
        f"Based on the diagnostic report, propose modifications to improve the skill library. "
        f"Focus on:\n"
        f"1. Fixing HARMFUL skills (negative Δ in correlation) — rewrite or remove\n"
        f"2. Fixing MISMATCH skills — sphere position is right but content is wrong\n"
        f"3. Merging redundant pairs identified by sphere analysis\n"
        f"4. Filling coverage gaps for failure patterns\n"
        f"5. Improving low-scoring but applicable skills\n"
        f"6. {'Adjusting training skill reward weights/data selection based on current stage' if not training_skill_locked else '(Training skills locked this round)'}\n\n"
        f"Return your proposal as a JSON object."
    )

    candidates = []
    for i in range(n_candidates):
        print(f"  Generating candidate {i+1}/{n_candidates}...")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": EVOLUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7 + i * 0.1,  # vary temperature across candidates
                max_completion_tokens=16384,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            candidate = json.loads(content)
            candidate["candidate_id"] = i + 1
            candidate["temperature"] = 0.7 + i * 0.1
            candidate["training_skill_locked"] = training_skill_locked

            # Backward compat: support old "modifications" key as "behavioral_modifications"
            if "modifications" in candidate and "behavioral_modifications" not in candidate:
                candidate["behavioral_modifications"] = candidate.pop("modifications")
            if "training_modifications" not in candidate:
                candidate["training_modifications"] = []

            n_beh = len(candidate.get("behavioral_modifications", []))
            n_train = len(candidate.get("training_modifications", []))
            print(f"    → {n_beh} behavioral + {n_train} training modifications proposed")
            candidates.append(candidate)
        except Exception as e:
            print(f"    → Error: {e}")

    return candidates


def apply_training_modifications(training_skills: dict, modifications: list[dict]) -> dict:
    """Apply training skill modifications (returns new copy)."""
    new_ts = copy.deepcopy(training_skills)
    ts_list = new_ts.get("training_skills", [])
    ts_by_id = {ts["skill_id"]: ts for ts in ts_list}

    for mod in modifications:
        action = mod.get("action", "")

        if action == "modify_training":
            sid = mod.get("skill_id", "")
            changes = mod.get("changes", {})
            if sid in ts_by_id:
                ts_by_id[sid].update(changes)

        elif action == "add_training":
            new_skill = mod.get("skill", {})
            if new_skill.get("skill_id"):
                ts_list.append(new_skill)
                ts_by_id[new_skill["skill_id"]] = new_skill

        elif action == "switch_training":
            sid = mod.get("skill_id", "")
            if sid in ts_by_id:
                new_ts["active_skill_id"] = sid

    new_ts["training_skills"] = ts_list
    return new_ts


def apply_modifications(skills: dict, modifications: list[dict]) -> dict:
    """Apply a candidate's behavioral modifications to the skill library (returns new copy)."""
    new_skills = copy.deepcopy(skills)

    # Build skill_id index
    id_to_location = {}
    for i, s in enumerate(new_skills.get("general_skills", [])):
        sid = s.get("skill_id", f"gen_{i+1:03d}")
        id_to_location[sid] = ("general_skills", i)
    for cat, cat_skills in new_skills.get("task_specific_skills", {}).items():
        for i, s in enumerate(cat_skills):
            sid = s.get("skill_id", f"{cat}_{i+1:03d}")
            id_to_location[sid] = ("task_specific_skills", cat, i)
    for i, s in enumerate(new_skills.get("common_mistakes", [])):
        sid = s.get("mistake_id", f"cm_{i+1:03d}")
        id_to_location[sid] = ("common_mistakes", i)

    remove_ids = set()

    for mod in modifications:
        action = mod.get("action", "")

        if action == "modify":
            sid = mod.get("skill_id", "")
            changes = mod.get("changes", {})
            loc = id_to_location.get(sid)
            if loc and loc[0] == "general_skills":
                skill = new_skills["general_skills"][loc[1]]
                skill.update(changes)
            elif loc and loc[0] == "task_specific_skills":
                skill = new_skills["task_specific_skills"][loc[1]][loc[2]]
                skill.update(changes)

        elif action == "add":
            new_skill = mod.get("skill", {})
            # Auto-detect scoring_type if not provided
            if "scoring_type" not in new_skill:
                scoring_text = new_skill.get("scoring", "").lower()
                prog_kw = ["count the number", "count total", "count steps",
                           "calls <=", "calls >=", "check if apis."]
                new_skill["scoring_type"] = (
                    "programmatic" if any(k in scoring_text for k in prog_kw)
                    else "llm"
                )
            cat = new_skill.get("category", "general")
            if cat == "general" or cat == "common_mistakes":
                new_skill["skill_id"] = f"gen_{len(new_skills['general_skills'])+1:03d}"
                new_skills["general_skills"].append(new_skill)
            elif "/" in cat:
                subcat = cat.split("/", 1)[1]
                if subcat not in new_skills.get("task_specific_skills", {}):
                    new_skills["task_specific_skills"][subcat] = []
                new_skill["skill_id"] = f"{subcat}_{len(new_skills['task_specific_skills'][subcat])+1:03d}"
                new_skills["task_specific_skills"][subcat].append(new_skill)

        elif action == "remove":
            remove_ids.add(mod.get("skill_id", ""))

        elif action == "merge":
            for sid in mod.get("skill_ids", []):
                remove_ids.add(sid)
            merged = mod.get("merged_skill", {})
            if merged:
                cat = merged.get("category", "general")
                if cat == "general":
                    merged["skill_id"] = f"gen_{len(new_skills['general_skills'])+1:03d}"
                    new_skills["general_skills"].append(merged)
                elif "/" in cat:
                    subcat = cat.split("/", 1)[1]
                    if subcat not in new_skills.get("task_specific_skills", {}):
                        new_skills["task_specific_skills"][subcat] = []
                    merged["skill_id"] = f"{subcat}_{len(new_skills['task_specific_skills'][subcat])+1:03d}"
                    new_skills["task_specific_skills"][subcat].append(merged)

    # Remove marked skills
    if remove_ids:
        new_skills["general_skills"] = [
            s for s in new_skills["general_skills"]
            if s.get("skill_id") not in remove_ids
        ]
        for cat in list(new_skills.get("task_specific_skills", {}).keys()):
            new_skills["task_specific_skills"][cat] = [
                s for s in new_skills["task_specific_skills"][cat]
                if s.get("skill_id") not in remove_ids
            ]
        new_skills["common_mistakes"] = [
            s for s in new_skills["common_mistakes"]
            if s.get("mistake_id") not in remove_ids
        ]

    return new_skills


def main():
    parser = argparse.ArgumentParser(description="ATS Skill Evolution (Outer Loop)")
    parser.add_argument("--skills_path", type=str,
                        default="data/skills/appworld_skills_ats.json")
    parser.add_argument("--diagnostics", type=str,
                        default="results/diagnostics/appworld_report.json")
    parser.add_argument("--training_skills", type=str,
                        default="data/skills/training_skills.json")
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Number of candidate proposals to generate")
    parser.add_argument("--output_dir", type=str,
                        default="results/evolution/")
    parser.add_argument("--apply_best", action="store_true",
                        help="Apply best candidate and save modified skills")
    parser.add_argument("--training_skill_locked", action="store_true",
                        help="Lock training skills (behavioral only this round)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading skill library...")
    skills = load_skills(args.skills_path)
    total = (len(skills.get("general_skills", []))
             + sum(len(v) for v in skills.get("task_specific_skills", {}).values())
             + len(skills.get("common_mistakes", [])))
    print(f"  {total} skills loaded")

    print("Loading diagnostics...")
    diagnostics = load_diagnostics(args.diagnostics)

    print("Loading training skills...")
    training_skills = load_training_skills(args.training_skills)
    active = training_skills.get("active_skill_id", "?")
    print(f"  Active training skill: {active}")

    print(f"\nGenerating {args.n_candidates} candidates...")
    print(f"  Training skill locked: {args.training_skill_locked}")
    client = AzureOpenAI(**AZURE_CONFIG)
    candidates = generate_candidates(
        client, skills, diagnostics, training_skills, args.n_candidates,
        training_skill_locked=args.training_skill_locked,
    )

    # Save candidates
    for c in candidates:
        cid = c["candidate_id"]
        path = os.path.join(args.output_dir, f"candidate_{cid}.json")
        with open(path, "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
        print(f"\nCandidate {cid} saved to {path}")
        print(f"  Reasoning: {c.get('reasoning', '?')[:200]}")
        beh_mods = c.get("behavioral_modifications", [])
        train_mods = c.get("training_modifications", [])
        actions = {}
        for m in beh_mods + train_mods:
            a = m.get("action", "?")
            actions[a] = actions.get(a, 0) + 1
        print(f"  Actions: {dict(actions)}")

    # Apply each candidate and save modified skill libraries + training skills
    for c in candidates:
        cid = c["candidate_id"]
        beh_mods = c.get("behavioral_modifications", [])
        train_mods = c.get("training_modifications", [])

        # Apply behavioral modifications
        if beh_mods:
            new_skills = apply_modifications(skills, beh_mods)
        else:
            new_skills = copy.deepcopy(skills)

        new_total = (len(new_skills.get("general_skills", []))
                     + sum(len(v) for v in new_skills.get("task_specific_skills", {}).values())
                     + len(new_skills.get("common_mistakes", [])))
        path = os.path.join(args.output_dir, f"skills_candidate_{cid}.json")
        with open(path, "w") as f:
            json.dump(new_skills, f, indent=2, ensure_ascii=False)
        print(f"  Candidate {cid}: {total} → {new_total} behavioral skills → {path}")

        # Apply training modifications
        if train_mods:
            new_ts = apply_training_modifications(training_skills, train_mods)
            ts_path = os.path.join(args.output_dir, f"training_skills_candidate_{cid}.json")
            with open(ts_path, "w") as f:
                json.dump(new_ts, f, indent=2, ensure_ascii=False)
            print(f"  Candidate {cid}: training skills → {ts_path} (active: {new_ts.get('active_skill_id', '?')})")

    print("\n" + "=" * 60)
    print("Evolution complete. Next steps:")
    print("  1. Review candidates in results/evolution/")
    print("  2. Run proxy eval with each candidate's skill set")
    print("  3. Pick best candidate by proxy reward")
    print("  4. (Optional) --apply_best to update the skill library")


if __name__ == "__main__":
    main()
