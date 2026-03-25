#!/usr/bin/env python3
"""End-to-end evaluation: build sphere → run agent on AppWorld → report results.

Usage:
    # No skills (baseline)
    python scripts/evaluate_appworld.py --mode none --backend vllm --model Qwen/Qwen3-8B

    # Sphere retrieval (our method)
    python scripts/evaluate_appworld.py --mode sphere --backend vllm --model Qwen/Qwen3-8B

    # Embedding top-K baseline
    python scripts/evaluate_appworld.py --mode embed_topk --backend vllm --model Qwen/Qwen3-8B

    # Limit episodes
    python scripts/evaluate_appworld.py --mode none --n_episodes 5 --backend vllm --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from skill_sphere.agent.llm_client import LLMClient
from skill_sphere.agent.appworld_agent import AppWorldAgent


def build_sphere(skills_path: str, device: str = "cpu"):
    """Build SkillSphere from skills JSON."""
    from skill_sphere.skill_bank.encoder import SkillEncoder
    from skill_sphere.skill_bank.skill_sphere import SkillSphere

    print(f"Loading embedding model...")
    encoder = SkillEncoder(device=device)
    print(f"Building sphere from: {skills_path}")
    sphere = SkillSphere.from_skillrl_json(skills_path, encoder=encoder, device=device)

    summary = sphere.summary()
    print(f"Sphere built: {summary['total_skills']} skills, categories={summary['categories']}")
    if "coverage_uniformity" in summary:
        print(f"  Coverage uniformity: {summary['coverage_uniformity']:.3f}")
        print(f"  Avg pairwise distance: {summary['avg_pairwise_distance']:.3f}")
    return sphere


def run_appworld_evaluation(
    agent: AppWorldAgent,
    n_episodes: int | None = None,
    max_steps: int = 30,
    split: str = "test_normal",
    experiment_name: str = "sphere_eval",
    save_trajectories: str | None = None,
    model_name: str = "unknown",
    mode: str = "none",
    task_file: str | None = None,
) -> tuple[dict, list[dict]]:
    """Run evaluation on AppWorld.

    Args:
        agent: Configured AppWorldAgent.
        n_episodes: Number of episodes to run. None = all available.
        max_steps: Maximum steps per episode.
        split: AppWorld dataset split.
        experiment_name: Name for AppWorld's output directory.
        save_trajectories: Path for trajectory JSONL (full prompts + sphere signals).
        model_name: For trajectory metadata.
        mode: For trajectory metadata.

    Returns:
        (results_dict, trajectory_logs)
    """
    from skill_sphere.env.appworld_wrapper import AppWorldEnv

    env = AppWorldEnv(
        experiment_name=experiment_name,
        max_interactions=max_steps,
    )

    # Load task IDs: from file (data selection) or from split
    if task_file and os.path.exists(task_file):
        with open(task_file) as f:
            task_ids = [line.strip() for line in f if line.strip()]
        print(f"Running {len(task_ids)} tasks from task_file '{task_file}'")
    else:
        task_ids = env.get_task_ids(split)
        if n_episodes is not None:
            task_ids = task_ids[:n_episodes]
        print(f"Running {len(task_ids)} tasks from split '{split}'")

    # Checkpoint: load completed tasks from existing trajectory file
    results_list: list[dict] = []
    all_logs: list[dict] = []
    completed_task_ids: set[str] = set()

    if save_trajectories and Path(save_trajectories).exists():
        with open(save_trajectories) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                all_logs.append(entry)
                completed_task_ids.add(entry["task_id"])
                results_list.append({
                    "task_id": entry["task_id"],
                    "success": entry.get("success", False),
                    "tgc": entry.get("tgc", 0),
                    "sgc": entry.get("sgc", 0),
                    "pass_count": entry.get("evaluation", {}).get("pass_count", 0),
                    "fail_count": entry.get("evaluation", {}).get("fail_count", 0),
                    "n_steps": entry.get("n_steps", 0),
                    "elapsed_seconds": entry.get("metadata", {}).get("elapsed_seconds", 0),
                })
        if completed_task_ids:
            succ = sum(1 for r in results_list if r["success"])
            print(f"Resuming from checkpoint: {len(completed_task_ids)} tasks done ({succ} successes)")

    traj_file = None
    if save_trajectories:
        Path(save_trajectories).parent.mkdir(parents=True, exist_ok=True)
        traj_file = open(save_trajectories, "a")  # append mode for resume

    for episode_idx, task_id in enumerate(task_ids):
        # Skip already-completed tasks (checkpoint resume)
        if task_id in completed_task_ids:
            continue

        try:
            init_result = env.reset(task_id)
        except Exception as e:
            print(f"  [{episode_idx+1}/{len(task_ids)}] {task_id}: RESET FAILED - {e}")
            results_list.append({"task_id": task_id, "success": False, "error": str(e)})
            continue

        task_desc = init_result.instruction
        print(f"\n[{episode_idx+1}/{len(task_ids)}] {task_id}: {task_desc[:80]}...")
        print(f"  Apps: {', '.join(init_result.allowed_apps)}")

        episode_start = time.time()

        def step_fn(code):
            step_result = env.step(code)
            return step_result.observation, step_result.task_completed

        try:
            success, logs = agent.run_episode(
                task_description=task_desc,
                supervisor_info=init_result.supervisor,
                allowed_apps=init_result.allowed_apps,
                step_fn=step_fn,
                max_steps=max_steps,
            )
        except Exception as e:
            print(f"  Episode failed with error: {e}")
            success = False
            logs = []

        # Ground-truth evaluation
        eval_success = False
        try:
            eval_result = env.evaluate()
            eval_success = eval_result.success
            tgc = 1.0 if eval_success else 0.0
            sgc = eval_result.pass_percentage / 100.0
            pass_count = eval_result.pass_count
            fail_count = eval_result.fail_count
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            tgc = 0.0
            sgc = 0.0
            pass_count = 0
            fail_count = 0

        elapsed = time.time() - episode_start
        n_steps = len(logs)
        status = "SUCCESS" if eval_success else "FAIL"
        print(f"  {status} in {n_steps} steps (TGC={tgc:.0f}, SGC={sgc:.1%}, "
              f"pass={pass_count}, fail={fail_count}, {elapsed:.1f}s)")

        result = {
            "task_id": task_id,
            "success": eval_success,
            "tgc": tgc,
            "sgc": sgc,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "n_steps": n_steps,
            "elapsed_seconds": elapsed,
        }
        results_list.append(result)

        # Build trajectory record (with full prompts for GRPO training)
        step_data = []
        for log in logs:
            sd = {
                "step": log.step,
                "action": log.action,
                "observation": log.observation[:500],
                "thinking": log.thinking[:500],
                "action_valid": log.action_valid,
                "reward": log.reward,
                "confidence": log.confidence,
                "system_prompt": log.system_prompt,
                "user_prompt": log.user_prompt,
                "raw_output": log.raw_output,
            }
            if log.skills_injected:
                sd["sphere_signals"] = {
                    "skills_injected": log.skills_injected,
                    "skills_used": log.skills_used,
                    "skill_scores": log.skill_scores,
                    "injection_strength": log.injection_strength,
                    "drift_rate": log.drift_rate,
                    "drift_norm": log.drift_norm,
                    "adaptive_momentum": log.adaptive_momentum,
                    "isolation_score": log.isolation_score,
                    "in_uncharted": log.in_uncharted,
                    "gamma": log.gamma,
                    "regime": log.regime,
                    "alignment": log.alignment,
                }
            step_data.append(sd)

        trajectory = {
            "environment": "appworld",
            "episode_id": episode_idx,
            "task_id": task_id,
            "task_description": task_desc,
            "success": result["success"],
            "tgc": tgc,
            "sgc": sgc,
            "n_steps": n_steps,
            "max_steps": max_steps,
            "model": model_name,
            "mode": mode,
            "trajectory": step_data,
            "evaluation": {
                "tgc": tgc,
                "sgc": sgc,
                "pass_count": pass_count,
                "fail_count": fail_count,
            },
            "metadata": {
                "allowed_apps": init_result.allowed_apps,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
            },
        }
        all_logs.append(trajectory)

        # Write trajectory incrementally
        if traj_file:
            traj_file.write(json.dumps(trajectory) + "\n")
            traj_file.flush()

    if traj_file:
        traj_file.close()

    env.close()

    # Compute aggregate results
    total = len(results_list)
    total_success = sum(1 for r in results_list if r.get("success", False))
    total_tgc = sum(r.get("tgc", 0) for r in results_list)
    total_sgc = sum(r.get("sgc", 0) for r in results_list)

    total_steps = sum(r.get("n_steps", 0) for r in results_list)
    results = {
        "tgc": total_tgc / total if total > 0 else 0.0,
        "sgc": total_sgc / total if total > 0 else 0.0,
        "success_rate": total_success / total if total > 0 else 0.0,
        "total_episodes": total,
        "total_successes": total_success,
        "avg_steps": total_steps / total if total > 0 else 0.0,
        "per_task": results_list,
    }

    return results, all_logs


def print_results(results: dict, mode: str):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(f"APPWORLD EVALUATION RESULTS (mode={mode})")
    print("=" * 60)
    print(f"TGC:  {results['tgc']:.1%} ({results['total_successes']}/{results['total_episodes']})")
    print(f"SGC:  {results['sgc']:.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Skill Sphere on AppWorld")
    parser.add_argument("--mode", type=str, default="none",
                        choices=["sphere", "embed_topk", "none"],
                        help="Skill retrieval mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="LLM model name")
    parser.add_argument("--skills_path", type=str,
                        default="data/skills/appworld_skills.json",
                        help="Path to skills JSON")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Number of episodes (None = all)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum interactions per task")
    parser.add_argument("--split", type=str, default="test_normal",
                        choices=["train", "dev", "test_normal", "test_challenge"],
                        help="AppWorld dataset split")
    parser.add_argument("--max_history", type=int, default=5,
                        help="Max history steps in prompt")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--save_trajectories", type=str, default=None,
                        help="Save trajectory JSONL (with full prompts)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for embedding model")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["azure", "vllm"],
                        help="LLM backend")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="AppWorld experiment name (default: auto-generated)")
    parser.add_argument("--max_new_tokens", type=int, default=3072,
                        help="Maximum tokens to generate")
    # Thinking mode
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable Qwen3 internal thinking mode")
    parser.add_argument("--thinking_budget", type=int, default=1024,
                        help="Max thinking tokens for Qwen3 (default 1024)")
    # Sphere parameters
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--min_inject_strength", type=float, default=0.15)
    parser.add_argument("--confidence_temperature", type=float, default=4.0)
    # Bridge parameters
    parser.add_argument("--enable_bridge", action="store_true",
                        help="Enable bridge skill selection")
    parser.add_argument("--bridge_k", type=int, default=1,
                        help="Number of bridge skills to inject")
    parser.add_argument("--task_file", type=str, default=None,
                        help="File with specific task IDs to run (one per line)")
    args = parser.parse_args()

    # Auto-generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"sphere_{args.mode}_{timestamp}"

    # Auto-generate trajectory path
    if args.save_trajectories is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        args.save_trajectories = f"data/trajectories/appworld/appworld_{args.mode}_{model_short}.jsonl"

    # Auto-generate output path
    if args.output is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        args.output = f"results/appworld_{args.mode}_{model_short}.json"

    # Build LLM client
    enable_thinking = None if not args.no_thinking else False
    thinking_budget = args.thinking_budget if not args.no_thinking else None
    llm = LLMClient(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        vllm_base_url=args.vllm_url,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )

    # Build SkillSphere (if needed)
    sphere = None
    if args.mode != "none" and Path(args.skills_path).exists():
        sphere = build_sphere(args.skills_path, device=args.device)

    # Build agent
    agent = AppWorldAgent(
        llm=llm,
        skill_sphere=sphere,
        mode=args.mode,
        max_history=args.max_history,
        sigma=args.sigma,
        min_inject_strength=args.min_inject_strength,
        confidence_temperature=args.confidence_temperature,
        enable_bridge=args.enable_bridge,
        bridge_k=args.bridge_k,
    )

    # Run evaluation
    print(f"\nStarting AppWorld evaluation: mode={args.mode}, model={args.model}")
    print(f"Split: {args.split}, max_steps={args.max_steps}")
    start_time = time.time()

    results, trajectory_logs = run_appworld_evaluation(
        agent,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        split=args.split,
        experiment_name=args.experiment_name,
        save_trajectories=args.save_trajectories,
        model_name=args.model,
        mode=args.mode,
        task_file=args.task_file,
    )

    elapsed = time.time() - start_time
    results["mode"] = args.mode
    results["model"] = args.model
    results["split"] = args.split
    results["max_steps"] = args.max_steps
    results["max_history"] = args.max_history
    results["elapsed_seconds"] = elapsed

    print_results(results, args.mode)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {args.output}")

    if args.save_trajectories:
        print(f"Trajectories saved to: {args.save_trajectories} ({len(trajectory_logs)} episodes)")


if __name__ == "__main__":
    main()
