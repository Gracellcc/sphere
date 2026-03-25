"""
ATS + Sphere Training Pipeline (Professional Version)

Orchestrates:
  Inner loop: veRL GRPO (scripts/verl/run_ats_grpo.sh)
    - veRL handles: multi-turn rollout → reward → GRPO advantage → policy update
    - Our modules: env_appworld_ats.py (Sphere + AppWorld), reward_manager_ats.py (verifier)
    - Training Skills control: reward weights, data selection policy
  Outer loop: Skill evolution (scripts/verl/ats_outer_loop.py)
    - Every M inner epochs: diagnostics → candidate generation → proxy eval → update skills
    - Training Skills updated every K*M epochs
  Phase transition:
    Phase 1: API-driven evolution (GPT-5.4) + veRL GRPO
    Phase 2: Model-driven evolution (same LoRA) + veRL GRPO
      - J(θ) = J_inner(θ) + λ_outer · J_outer(θ)

Hardware: 4× A6000 48GB
Base model: Qwen3-8B (or SFT warmup checkpoint)

Usage:
  # Dry run: simulate pipeline without execution
  python scripts/ats_pipeline.py --dry_run --n_outer_steps 2

  # Full GRPO (Phase 1, auto Phase 2 transition)
  python scripts/ats_pipeline.py --mode grpo --n_outer_steps 5 --n_inner_epochs 10

  # Debug: minimal run to verify pipeline
  python scripts/ats_pipeline.py --mode grpo --n_outer_steps 1 --n_inner_epochs 2 --debug
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT_DIR = str(Path(__file__).resolve().parent.parent)


# =============================================================================
# Training State: tracks progress + selects Training Skill
# =============================================================================

class TrainingState:
    """Tracks training progress and statistics across outer iterations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.outer_step = 0
        self.total_epochs = 0
        self.total_episodes = 0
        self.success_count = 0
        self.type_stats = defaultdict(lambda: {"total": 0, "success": 0})
        self.history = []  # (outer_step, success_rate, skills_version)

        self.state_path = os.path.join(output_dir, "training_state.json")
        if os.path.exists(self.state_path):
            self._load()

    def _load(self):
        with open(self.state_path) as f:
            d = json.load(f)
        self.outer_step = d.get("outer_step", 0)
        self.total_epochs = d.get("total_epochs", 0)
        self.total_episodes = d.get("total_episodes", 0)
        self.success_count = d.get("success_count", 0)
        self.type_stats = defaultdict(
            lambda: {"total": 0, "success": 0}, d.get("type_stats", {}))
        self.history = d.get("history", [])

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump({
                "outer_step": self.outer_step,
                "total_epochs": self.total_epochs,
                "total_episodes": self.total_episodes,
                "success_count": self.success_count,
                "success_rate": self.success_rate,
                "type_stats": dict(self.type_stats),
                "history": self.history,
            }, f, indent=2)

    @property
    def success_rate(self):
        return self.success_count / self.total_episodes if self.total_episodes > 0 else 0

    def select_training_skill(self, training_skills: list[dict]) -> dict:
        """Select active training skill based on current success rate.

        Priority: iterate all skills, the LAST one whose condition matches wins.
        This gives higher-threshold skills priority when multiple match.

        Supports compound conditions via "and" nested key:
          {"metric": "overall_success_rate", "op": ">", "value": 0.2,
           "and": {"metric": "weakest_type_success_rate", "op": "<", "value": 0.1}}
        """
        if not training_skills:
            return {"title": "default", "reward_formula": "r = 1.0 × outcome + 0.3 × supervision"}

        selected = training_skills[0]  # fallback: first skill

        for ts in training_skills:
            cond = ts.get("when_to_use_condition", {})
            if self._eval_condition(cond):
                selected = ts

        return selected

    def _eval_condition(self, cond: dict) -> bool:
        """Evaluate a single condition (with optional 'and' clause)."""
        if not cond:
            return True

        metric = cond.get("metric", "")
        op = cond.get("op", "")
        value = cond.get("value", 0)

        # Resolve metric value
        metric_val = self._get_metric_value(metric)
        if metric_val is None:
            return False

        # Check primary condition
        if not self._check_op(metric_val, op, value):
            return False

        # Check compound "and" condition
        and_cond = cond.get("and")
        if and_cond:
            return self._eval_condition(and_cond)

        return True

    def _get_metric_value(self, metric: str):
        """Resolve a metric name to its current value."""
        if metric == "overall_success_rate":
            return self.success_rate
        elif metric == "weakest_type_success_rate":
            if not self.type_stats:
                return 0.0
            type_rates = []
            for type_name, stats in self.type_stats.items():
                total = stats.get("total", 0)
                if total > 0:
                    type_rates.append(stats.get("success", 0) / total)
            return min(type_rates) if type_rates else 0.0
        return None

    @staticmethod
    def _check_op(val: float, op: str, threshold: float) -> bool:
        if op == "<":
            return val < threshold
        elif op == ">":
            return val > threshold
        elif op == ">=":
            return val >= threshold
        elif op == "<=":
            return val <= threshold
        return False

    def update_from_verl_metrics(self, metrics: dict):
        """Update state from veRL's logged metrics (parsed from console output)."""
        if "success_rate" in metrics:
            # Approximate episode count from veRL batch
            n_episodes = metrics.get("n_episodes", 0)
            n_successes = int(metrics.get("success_rate", 0) * n_episodes)
            self.total_episodes += n_episodes
            self.success_count += n_successes

    def update_type_stats_from_file(self, task_stats_path: str):
        """Update per-type statistics from task_stats.json.

        This enables compound conditions like 'weakest_type_success_rate < 10%'.
        task_stats.json format: {task_id: {"total": N, "success": M, "type": "..."}, ...}
        """
        if not task_stats_path or not os.path.exists(task_stats_path):
            return
        try:
            with open(task_stats_path) as f:
                task_stats = json.load(f)
            # Aggregate per task-type
            type_agg = defaultdict(lambda: {"total": 0, "success": 0})
            for task_id, stats in task_stats.items():
                task_type = stats.get("type", "unknown")
                type_agg[task_type]["total"] += stats.get("total", 0)
                type_agg[task_type]["success"] += stats.get("success", 0)
            self.type_stats = type_agg
        except Exception as e:
            print(f"[TrainingState] Failed to update type_stats: {e}")


# =============================================================================
# Reward Weight Parser
# =============================================================================

def parse_reward_formula(formula: str) -> dict:
    """Parse reward formula string into weights dict.

    Supports formats like:
      'r = 1.0 × outcome + 0.3 × supervision'
      'r = 0.5 × outcome + 1.0 × supervision + 0.3 × (1 - steps/max_steps)'
    """
    weights = {"outcome": 0.0, "supervision": 0.0, "efficiency": 0.0}

    for match in re.finditer(r'([\d.]+)\s*[*×]\s*(outcome|supervision)', formula):
        w, name = float(match.group(1)), match.group(2)
        weights[name] = w

    eff_match = re.search(r'([\d.]+)\s*[*×]\s*\(1\s*-\s*steps', formula)
    if eff_match:
        weights["efficiency"] = float(eff_match.group(1))

    return weights


def training_skill_to_data_selection(ts: dict) -> str:
    """Map training skill's data_selection text to dataset policy keyword."""
    ds = ts.get("data_selection", "").lower()
    if "成功率最低" in ds or "weakness" in ds:
        return "weakness_focused"
    elif "成功率>30%" in ds or "efficiency" in ds:
        return "efficiency_push"
    else:
        return "balanced"


# =============================================================================
# ATS Pipeline: Orchestrator
# =============================================================================

class ATSPipeline:
    """Orchestrates veRL GRPO (inner) + skill evolution (outer) + Training Skills."""

    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Paths
        self.skills_dir = os.path.join(self.output_dir, "skills_versions")
        self.verl_output_dir = os.path.join(self.output_dir, "verl_checkpoints")
        self.evolution_dir = os.path.join(self.output_dir, "evolution")
        self.dev_eval_dir = os.path.join(self.output_dir, "dev_evals")
        for d in [self.skills_dir, self.verl_output_dir,
                  self.evolution_dir, self.dev_eval_dir]:
            os.makedirs(d, exist_ok=True)

        # Current state
        self.current_skills = args.skills_path
        shutil.copy2(self.current_skills,
                     os.path.join(self.skills_dir, "skills_v0.json"))
        self.current_model_path = args.model
        self.state = TrainingState(self.output_dir)
        self.log_path = os.path.join(self.output_dir, "pipeline.log")

        # Phase tracking
        self._phase = 1
        self._evolution_data_count = 0
        self._dev_tgc_history = []

        # Phase 2 transition data: accumulate (diagnostics, candidate, proxy_reward) triplets
        self._evolution_sft_path = os.path.join(self.output_dir, "evolution_sft_data.jsonl")
        self._model_candidate_ratio = 0.0  # 0.0 = all API, 1.0 = all model

        # vLLM server process (started for outer loop proxy eval, stopped after)
        self._vllm_proc = None
        self._vllm_port = 8000

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    # =========================================================================
    # vLLM Server Lifecycle (for proxy eval in outer loop)
    # =========================================================================

    def _start_vllm_server(self) -> bool:
        """Start a vLLM OpenAI-compatible server for proxy eval.

        Runs on GPU 0 after veRL GRPO releases all GPUs.
        If current_model_path is a LoRA adapter, loads base model + LoRA.
        Returns True if server is healthy.
        """
        model_path = self.current_model_path
        port = self._vllm_port
        vllm_log = os.path.join(self.output_dir, "vllm_proxy_eval.log")

        # Detect if model_path is a LoRA adapter (has adapter_config.json but no config.json)
        adapter_config = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config) and not os.path.exists(
            os.path.join(model_path, "config.json"))

        if is_lora:
            # Read base model path from adapter config
            import json as _json
            with open(adapter_config) as f:
                base_model = _json.load(f)["base_model_name_or_path"]
            self.log(f"  Starting vLLM server: base={base_model}, lora={model_path}, port={port}")
        else:
            base_model = model_path
            self.log(f"  Starting vLLM server: model={model_path}, port={port}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", base_model,
            "--port", str(port),
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.85",
            "--dtype", "bfloat16",
        ]
        if is_lora:
            cmd.extend(["--enable-lora", "--lora-modules",
                         f"lora_adapter={model_path}"])

        self._vllm_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(vllm_log, "w"),
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready (up to 120s)
        import urllib.request
        for i in range(24):
            time.sleep(5)
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{port}/v1/models", timeout=3)
                if resp.status == 200:
                    self.log(f"  vLLM server ready after {(i + 1) * 5}s")
                    return True
            except Exception:
                pass
            # Check if process died
            if self._vllm_proc.poll() is not None:
                self.log(f"  vLLM server died (rc={self._vllm_proc.returncode})")
                self._vllm_proc = None
                return False

        self.log(f"  vLLM server failed to start within 120s")
        self._stop_vllm_server()
        return False

    def _stop_vllm_server(self):
        """Stop the vLLM server process."""
        if self._vllm_proc is None:
            return
        self.log(f"  Stopping vLLM server (PID={self._vllm_proc.pid})")
        try:
            self._vllm_proc.terminate()
            try:
                self._vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
                self._vllm_proc.wait(timeout=5)
        except Exception as e:
            self.log(f"  Warning: failed to stop vLLM: {e}")
        self._vllm_proc = None

    def _cleanup_gpu_processes(self):
        """Clean up orphan ray/vLLM processes after GRPO failure to free GPU memory."""
        self.log(f"  Cleaning up GPU processes...")
        try:
            subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
        except Exception:
            pass
        # Kill any remaining vLLM/ray worker processes
        try:
            result = subprocess.run(
                ["bash", "-c",
                 "ps aux | grep -E 'ray::Worker|vllm.entrypoints' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null"],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
        time.sleep(3)  # Allow GPU memory to be released
        self.log(f"  GPU cleanup done.")

    # =========================================================================
    # Inner Loop: veRL GRPO
    # =========================================================================

    def run_verl_grpo(self, outer_step: int, training_skill: dict,
                      n_epochs: int) -> str:
        """Run veRL GRPO as inner loop for N epochs.

        Returns path to latest checkpoint, or empty string on failure.

        Training Skill controls veRL behavior via environment variables:
          - OUTCOME_WEIGHT, SUPERVISION_WEIGHT, EFFICIENCY_WEIGHT → reward manager
          - DATA_SELECTION → dataset's data selection policy
          - TASK_STATS_PATH → per-task stats for data selection
          - SKILLS_PATH → behavioral skills for Sphere
          - MODEL_PATH → current model checkpoint
        """
        # Get reward weights: prefer pre-parsed reward_weights, fall back to regex parsing
        weights = training_skill.get("reward_weights")
        if not weights:
            formula = training_skill.get("reward_formula", "r = 1.0 × outcome + 0.3 × supervision")
            weights = parse_reward_formula(formula)
        else:
            # Ensure all keys exist
            weights = {
                "outcome": weights.get("outcome", 1.0),
                "supervision": weights.get("supervision", 0.0),
                "efficiency": weights.get("efficiency", 0.0),
            }
        data_selection = training_skill_to_data_selection(training_skill)

        self.log(f"  veRL GRPO: {n_epochs} epochs")
        self.log(f"  Reward: outcome={weights['outcome']}, "
                 f"supervision={weights['supervision']}, "
                 f"efficiency={weights['efficiency']}")
        self.log(f"  Data selection: {data_selection}")
        self.log(f"  Model: {self.current_model_path}")
        self.log(f"  Skills: {self.current_skills}")

        # Determine veRL script
        if self.args.debug:
            script = os.path.join(ROOT_DIR, "scripts/verl/run_ats_grpo_debug.sh")
        elif self.args.quick:
            script = os.path.join(ROOT_DIR, "scripts/verl/run_ats_grpo_quick.sh")
        else:
            script = os.path.join(ROOT_DIR, "scripts/verl/run_ats_grpo.sh")

        # Experiment name: unique per outer step to avoid checkpoint conflicts
        experiment_name = f"ats_grpo_outer{outer_step}"
        ckpt_dir = os.path.join(
            self.verl_output_dir, experiment_name)

        # Build environment
        env = os.environ.copy()
        env.update({
            # Model
            "MODEL_PATH": self.current_model_path,
            # Reward weights (from Training Skill)
            "OUTCOME_WEIGHT": str(weights["outcome"]),
            "SUPERVISION_WEIGHT": str(weights["supervision"]),
            "EFFICIENCY_WEIGHT": str(weights["efficiency"]),
            # Skills
            "SKILLS_PATH": os.path.abspath(self.current_skills),
            # Data selection
            "TRAIN_DATA_SIZE": str(self.args.tasks_per_epoch),
            "VAL_DATA_SIZE": str(self.args.val_tasks),
            "GROUP_SIZE": str(self.args.G),
            "MAX_STEPS": str(self.args.max_steps),
            # Task stats for data selection policy
            "TASK_STATS_PATH": self._get_task_stats_path(),
            # Hardware
            "N_GPUS": str(self.args.n_gpus),
            "TP_SIZE": str(self.args.tp_size),
            # veRL overrides
            "VLLM_ATTENTION_BACKEND": env.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"),
        })

        # Pass dynamic parameters via environment variables (avoid Hydra duplicate key errors)
        env["TOTAL_EPOCHS"] = str(n_epochs)
        env["SAVE_FREQ"] = str(self.args.save_freq)
        env["TEST_FREQ"] = str(self.args.test_freq)
        env["EXPERIMENT_NAME"] = experiment_name
        env["DEFAULT_LOCAL_DIR"] = os.path.join(self.verl_output_dir, experiment_name)
        env["DATA_SELECTION"] = data_selection
        # Skip val_before_train (base model score already known from prior runs)
        env["VAL_BEFORE_TRAIN"] = "False"
        # LoRA (for memory-constrained hardware; set to 0 to disable)
        if self.args.lora_rank > 0:
            env["LORA_RANK"] = str(self.args.lora_rank)
            env["LORA_ALPHA"] = str(self.args.lora_rank)

        # Hardware-aware configuration
        is_large_gpu = self.args.n_gpus >= 4 and self.args.lora_rank == 0
        if is_large_gpu:
            # A100 80GB × 8: full finetune, no offload needed
            env["PARAM_OFFLOAD"] = "False"
            env["OPTIMIZER_OFFLOAD"] = "False"
            env["FREE_CACHE_ENGINE"] = "False"
            env["GPU_MEM_UTIL"] = str(getattr(self.args, 'gpu_mem_util', 0.6))
            env["LOG_PROB_MICRO_BS"] = str(getattr(self.args, 'log_prob_micro_bs', 16))
        else:
            # A6000 48GB: LoRA + offload
            env["PARAM_OFFLOAD"] = str(getattr(self.args, 'param_offload', True))
            env["OPTIMIZER_OFFLOAD"] = str(getattr(self.args, 'optimizer_offload', True))
            env["FREE_CACHE_ENGINE"] = str(getattr(self.args, 'free_cache_engine', True))
            env["GPU_MEM_UTIL"] = str(getattr(self.args, 'gpu_mem_util', 0.4))
            env["LOG_PROB_MICRO_BS"] = str(getattr(self.args, 'log_prob_micro_bs', 2))

        # GPU selection (use first N GPUs only if less than total)
        if self.args.n_gpus < 4:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.args.n_gpus))

        # Resume from previous outer step's checkpoint
        verl_overrides = []
        prev_ckpt = self._find_latest_checkpoint(outer_step - 1)
        if prev_ckpt:
            verl_overrides.append(f"trainer.resume_from_path={prev_ckpt}")
            self.log(f"  Resuming from: {prev_ckpt}")

        # Hydra overrides
        verl_overrides.append("+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16")
        if is_large_gpu:
            # Full finetune on large GPUs: save HF model for vLLM/dev_eval
            n_gpus = self.args.n_gpus
            verl_overrides.append(f"actor_rollout_ref.actor.ppo_mini_batch_size={n_gpus * 4}")
            verl_overrides.append(f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4")
            verl_overrides.append("actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]")
        else:
            # LoRA on small GPUs
            verl_overrides.append("actor_rollout_ref.actor.ppo_mini_batch_size=4")
            verl_overrides.append("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2")
            verl_overrides.append("actor_rollout_ref.rollout.enforce_eager=True")
            verl_overrides.append("actor_rollout_ref.rollout.load_format=safetensors")

        # First positional arg is ENGINE (vllm/sglang), then any Hydra overrides via "$@"
        cmd = ["bash", script, "vllm"] + verl_overrides

        if self.args.dry_run:
            self.log(f"  [DRY RUN] Would run veRL GRPO:")
            self.log(f"    {' '.join(cmd[-5:])}")
            self.log(f"    Env: OUTCOME_WEIGHT={weights['outcome']}, ...")
            return ""

        # Free GPU memory: stop vLLM server before veRL GRPO (veRL manages its own vLLM)
        self._stop_vllm_server()

        # Run veRL GRPO
        self.log(f"  Launching veRL GRPO...")
        verl_log = os.path.join(self.output_dir, f"verl_outer{outer_step}.log")
        with open(verl_log, "w") as logf:
            proc = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=self.args.verl_timeout, cwd=ROOT_DIR,
            )

        if proc.returncode != 0:
            self.log(f"  veRL GRPO failed (rc={proc.returncode})")
            # Show last 20 lines of log
            try:
                with open(verl_log) as f:
                    lines = f.readlines()
                for line in lines[-20:]:
                    self.log(f"    {line.rstrip()}")
            except Exception:
                pass
            # Clean up orphan ray/vLLM processes to free GPU memory
            self._cleanup_gpu_processes()
            return ""

        self.log(f"  veRL GRPO complete!")

        # Parse metrics from veRL log
        metrics = self._parse_verl_metrics(verl_log)
        if metrics:
            self.state.update_from_verl_metrics(metrics)
            self.log(f"  veRL metrics: success_rate={metrics.get('success_rate', 'N/A')}")

        # Find latest checkpoint
        latest_ckpt = self._find_latest_checkpoint(outer_step)
        if latest_ckpt:
            # For full finetune with hf_model, the HF-format model lives in
            # global_step_N/actor/huggingface/.  Use that for vLLM/dev_eval.
            # The raw global_step_N/ path is only used for veRL resume
            # (handled separately by _find_latest_checkpoint in the next outer step).
            hf_path = os.path.join(latest_ckpt, "actor", "huggingface")
            if os.path.isdir(hf_path) and os.path.exists(
                    os.path.join(hf_path, "config.json")):
                self.current_model_path = hf_path
                self.log(f"  Checkpoint (HF): {hf_path}")
            else:
                # LoRA or no hf_model saved — use raw checkpoint
                self.current_model_path = latest_ckpt
                self.log(f"  Checkpoint: {latest_ckpt}")
        else:
            self.log(f"  WARNING: No checkpoint found in {ckpt_dir}")

        self.state.total_epochs += n_epochs
        return latest_ckpt or ""

    def _find_latest_checkpoint(self, outer_step: int) -> str:
        """Find latest veRL checkpoint for a given outer step.

        veRL saves checkpoints as:
          {verl_output_dir}/{experiment_name}/global_step_{N}/
        """
        experiment = f"ats_grpo_outer{outer_step}"
        ckpt_base = os.path.join(self.verl_output_dir, experiment)

        if not os.path.exists(ckpt_base):
            return ""

        # Find highest global_step
        steps = []
        for d in os.listdir(ckpt_base):
            if d.startswith("global_step_"):
                try:
                    step = int(d.split("_")[-1])
                    steps.append(step)
                except ValueError:
                    pass

        if not steps:
            return ""

        latest = max(steps)
        return os.path.join(ckpt_base, f"global_step_{latest}")

    def _get_task_stats_path(self) -> str:
        """Get path to task-level statistics for data selection."""
        stats_path = os.path.join(self.output_dir, "task_stats.json")
        if os.path.exists(stats_path):
            return stats_path
        return ""

    def _parse_verl_metrics(self, log_path: str) -> dict:
        """Parse veRL console output for key metrics."""
        metrics = {}
        try:
            with open(log_path) as f:
                for line in f:
                    # veRL logs success_rate as a metric
                    if "success_rate" in line:
                        m = re.search(r'success_rate["\s:=]+([0-9.]+)', line)
                        if m:
                            metrics["success_rate"] = float(m.group(1))
                    # Episode count
                    if "n_episodes" in line or "total_episodes" in line:
                        m = re.search(r'(?:n_episodes|total_episodes)["\s:=]+(\d+)', line)
                        if m:
                            metrics["n_episodes"] = int(m.group(1))
        except Exception:
            pass
        return metrics

    def _save_task_stats(self, verl_log: str):
        """Extract per-task stats from veRL output and save for data selection."""
        # veRL's reward manager logs per-episode results
        # Parse these to build task-level stats for next iteration
        stats = {}
        try:
            with open(verl_log) as f:
                for line in f:
                    if "[ATS Reward]" in line and "task_id=" in line:
                        m = re.search(
                            r'task_id=(\S+)\s+success=(\d)\s+reward=([\d.]+)', line)
                        if m:
                            tid = m.group(1)
                            success = int(m.group(2))
                            if tid not in stats:
                                stats[tid] = {"total": 0, "success": 0, "success_rate": 0}
                            stats[tid]["total"] += 1
                            stats[tid]["success"] += success
                            stats[tid]["success_rate"] = (
                                stats[tid]["success"] / stats[tid]["total"])
        except Exception:
            pass

        if stats:
            # Merge with existing stats
            stats_path = os.path.join(self.output_dir, "task_stats.json")
            existing = {}
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    existing = json.load(f)
            existing.update(stats)
            with open(stats_path, "w") as f:
                json.dump(existing, f, indent=2)

    # =========================================================================
    # Outer Loop: Skill Evolution
    # =========================================================================

    def _save_evolution_triplet(self, outer_step: int, diagnostics_path: str,
                                candidate: dict, proxy_reward: float,
                                training_skill_locked: bool):
        """Accumulate (diagnostics, candidate, proxy_reward) for Phase 2 transition SFT.

        Design doc: "记录（诊断报告, candidate方案, proxy reward）数据"
        These triplets teach the model to generate skill modifications.
        """
        triplet = {
            "outer_step": outer_step,
            "diagnostics_path": diagnostics_path,
            "candidate": candidate,
            "proxy_reward": proxy_reward,
            "training_locked": training_skill_locked,
        }
        # Load diagnostics content for SFT
        if os.path.exists(diagnostics_path):
            with open(diagnostics_path) as f:
                triplet["diagnostics"] = json.load(f)

        with open(self._evolution_sft_path, "a") as f:
            f.write(json.dumps(triplet, ensure_ascii=False, default=str) + "\n")

        self._evolution_data_count += 1
        self.log(f"  Evolution SFT data: {self._evolution_data_count} triplets accumulated")

    def run_outer_evolution(self, outer_step: int, training_skills: list[dict],
                           training_skill_locked: bool) -> list[dict]:
        """Run outer loop: diagnostics → candidates → proxy eval → update skills.

        Design doc mapping:
          - Phase 1: API generates G candidates (each = 1 training + N behavioral)
          - Phase 2: Model generates candidates + API baselines
          - Both phases: proxy eval → select best → update active skills
          - Both phases: save evolution triplets for Phase 2 transition

        Uses ats_outer_loop.py which implements the full evolution pipeline.
        Returns updated training_skills (may be modified if unlocked).
        """
        self.log(f"\n{'='*60}")
        self.log(f"OUTER EVOLUTION (step {outer_step}, Phase {self._phase})")
        self.log(f"  Training skill locked: {training_skill_locked}")
        if self._phase >= 2:
            self.log(f"  Model candidate ratio: {self._model_candidate_ratio:.0%}")
        self.log(f"{'='*60}")

        # Use dev_eval trajectories for evolution diagnostics
        trajectory_dir = self.dev_eval_dir
        evo_output_dir = os.path.join(self.evolution_dir, f"outer_{outer_step}")

        cmd = [
            sys.executable, os.path.join(ROOT_DIR, "scripts/verl/ats_outer_loop.py"),
            "--skills_path", self.current_skills,
            "--training_skills_path", self.args.training_skills,
            "--trajectory_dir", trajectory_dir,
            "--step", str(outer_step),
            "--M", str(self.args.n_inner_epochs),
            "--K", str(self.args.training_skill_update_k),
            "--G", str(self.args.n_candidates),
            "--proxy_eval_tasks", str(self.args.proxy_n_tasks),
            "--sphere_path", self.current_skills,
            "--output_dir", evo_output_dir,
            "--verifier_model", self.args.verifier_model,
            "--proxy_model_path", self.current_model_path,
            "--model_candidate_ratio", str(self._model_candidate_ratio),
        ]

        if self.args.dry_run:
            self.log(f"  [DRY RUN] Would run outer evolution")
            return training_skills

        self.log(f"  Running outer evolution...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode != 0:
            self.log(f"  Evolution error: {proc.stderr[-300:]}")
            return training_skills

        if proc.stdout:
            self.log(f"  Evolution output (tail):")
            for line in proc.stdout.strip().split("\n")[-10:]:
                self.log(f"    {line}")

        # Save evolution triplets for Phase 2 transition
        # ats_outer_loop.py saves evolution_history.json with candidate + scores
        history_path = os.path.join(evo_output_dir, "evolution_history.json")
        diag_path = os.path.join(evo_output_dir,
                                  f"diagnostics_step{outer_step}.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            for record in (history if isinstance(history, list) else [history]):
                self._save_evolution_triplet(
                    outer_step, diag_path,
                    {"reasoning": record.get("reasoning", ""),
                     "changes": record.get("changes", [])},
                    record.get("best_score", 0.0),
                    training_skill_locked,
                )

        # Check if skills were updated
        if os.path.exists(self.current_skills):
            version = f"skills_v{outer_step + 1}.json"
            versioned = os.path.join(self.skills_dir, version)
            shutil.copy2(self.current_skills, versioned)
            self.log(f"  Skills saved: {version}")

        # Check if training skills were updated
        if not training_skill_locked and os.path.exists(self.args.training_skills):
            with open(self.args.training_skills) as f:
                training_skills = json.load(f)
                if isinstance(training_skills, dict):
                    training_skills = training_skills.get("training_skills", [training_skills])

        return training_skills

    # =========================================================================
    # Dev Eval
    # =========================================================================

    def run_dev_eval(self, outer_step: int):
        """Run held-out eval on dev split after model update."""
        self.log(f"\n--- Dev Eval (outer {outer_step}) ---")

        result_path = os.path.join(self.dev_eval_dir,
                                    f"dev_eval_outer_{outer_step}_result.json")
        traj_path = os.path.join(self.dev_eval_dir,
                                  f"dev_eval_outer_{outer_step}.jsonl")

        cmd = [
            sys.executable, os.path.join(ROOT_DIR, "scripts/evaluate_appworld.py"),
            "--mode", "sphere",
            "--skills_path", self.current_skills,
            "--split", "dev",
            "--backend", self.args.backend,
            "--model", self.current_model_path,
            "--output", result_path,
            "--save_trajectories", traj_path,
        ]
        if self.args.backend == "vllm":
            cmd.extend(["--vllm_url", self.args.vllm_url])
        # Default: no thinking (GRPO rollout can't limit thinking_budget,
        # so thinking risks eating all max_tokens and truncating code actions)
        if not self.args.enable_thinking:
            cmd.append("--no_thinking")

        if self.args.dry_run:
            self.log(f"  [DRY RUN] Would run dev eval")
            return

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if proc.returncode != 0:
            self.log(f"  Dev eval error: {proc.stderr[-300:]}")
            return

        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            tgc = results.get("tgc", 0)
            sgc = results.get("sgc", 0)
            n_eps = results.get("total_episodes", 0)
            n_succ = results.get("total_successes", 0)
            avg_steps = results.get("avg_steps", 0)

            self.log(f"  Dev TGC:  {tgc:.1%} ({n_succ}/{n_eps})")
            self.log(f"  Dev SGC:  {sgc:.1%}")
            self.log(f"  Dev Avg Steps: {avg_steps:.1f}")
            self._dev_tgc_history.append(tgc)

            # Save to history
            summary = {
                "outer_step": outer_step,
                "model": self.current_model_path,
                "skills": self.current_skills,
                "tgc": tgc, "sgc": sgc,
                "avg_steps": avg_steps,
                "n_episodes": n_eps, "n_successes": n_succ,
            }
            history_path = os.path.join(self.dev_eval_dir, "dev_eval_history.jsonl")
            with open(history_path, "a") as f:
                f.write(json.dumps(summary) + "\n")

    # =========================================================================
    # Phase 2 Transition
    # =========================================================================

    def _check_phase_transition(self) -> bool:
        """Check if Phase 1 → Phase 2 transition conditions are met."""
        if self._phase >= 2:
            return False

        data_threshold = getattr(self.args, 'phase2_data_threshold', 25)
        tgc_threshold = getattr(self.args, 'phase2_tgc_threshold', 0.15)

        if self._evolution_data_count < data_threshold:
            return False
        if not self._dev_tgc_history:
            return False
        if self._dev_tgc_history[-1] < tgc_threshold:
            return False

        self.log(f"\n*** Phase Transition Triggered ***")
        self.log(f"  Dev TGC: {self._dev_tgc_history[-1]:.1%} (> {tgc_threshold:.0%})")
        self.log(f"  Evolution data: {self._evolution_data_count} rounds (> {data_threshold})")
        return True

    def _execute_phase_transition(self):
        """Execute Phase 1 → Phase 2 transition.

        Design doc: "过渡SFT：用阶段1积累的（诊断报告, skill修改, 效果）数据
        SFT模型的skill修改能力"

        Steps:
          1. Run transition SFT on accumulated evolution triplets
          2. Switch to Phase 2 with gradual model candidate ratio (33% → 67% → 100%)
        """
        self.log(f"\n--- Transition SFT ---")

        if not os.path.exists(self._evolution_sft_path):
            self.log(f"  No evolution SFT data found, staying in Phase 1")
            return

        # Count triplets
        with open(self._evolution_sft_path) as f:
            n_triplets = sum(1 for line in f if line.strip())
        self.log(f"  Evolution SFT data: {n_triplets} triplets")

        # Run transition SFT (teaches model the Skill Evolver role)
        transition_dir = os.path.join(self.output_dir, "transition_sft")
        cmd = [
            sys.executable, os.path.join(ROOT_DIR, "scripts/train_transition_sft.py"),
            "--data_path", self._evolution_sft_path,
            "--base_model", self.current_model_path,
            "--output_dir", transition_dir,
            "--reward_threshold", "0.0",
            "--lora_r", "64",
            "--lora_alpha", "128",
            "--epochs", "2",
            "--lr", "1e-5",
        ]

        if self.args.dry_run:
            self.log(f"  [DRY RUN] Would run transition SFT")
            self._phase = 2
            self._model_candidate_ratio = 0.33
            return

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            self.log(f"  Transition SFT failed: {proc.stderr[-300:]}")
            self.log(f"  Staying in Phase 1")
            return

        # Update model path to include transition SFT
        if os.path.exists(transition_dir):
            self.current_model_path = transition_dir
            self.log(f"  Transition SFT complete: {transition_dir}")

        self._phase = 2
        self._model_candidate_ratio = 0.33  # start with 1/3 model candidates
        self.log(f"  Entered Phase 2 (model ratio: {self._model_candidate_ratio:.0%})")
        self.log(f"  Design: J(θ) = J_inner(θ) + λ_outer · J_outer(θ)")

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    def run(self):
        self.log(f"\n{'#'*60}")
        self.log(f"ATS Pipeline started")
        self.log(f"{'#'*60}")
        self.log(f"  Mode: {self.args.mode}")
        self.log(f"  Model: {self.args.model}")
        self.log(f"  Skills: {self.current_skills}")
        self.log(f"  Outer steps: {self.args.n_outer_steps}")
        self.log(f"  Inner epochs per outer: {self.args.n_inner_epochs}")
        self.log(f"  Tasks per epoch: {self.args.tasks_per_epoch}")
        self.log(f"  GRPO G (group size): {self.args.G}")
        self.log(f"  Max env steps: {self.args.max_steps}")
        self.log(f"  Training skill update freq: K={self.args.training_skill_update_k}")
        self.log(f"  Phase: {self._phase}")

        # Load training skills
        with open(self.args.training_skills) as f:
            training_skills_data = json.load(f)
        if isinstance(training_skills_data, dict):
            training_skills = training_skills_data.get("training_skills", [training_skills_data])
        elif isinstance(training_skills_data, list):
            training_skills = training_skills_data
        else:
            training_skills = [training_skills_data]

        self.log(f"  Training skills: {len(training_skills)}")
        for ts in training_skills:
            self.log(f"    - {ts.get('title', '?')}: {ts.get('reward_formula', 'N/A')[:60]}")

        for outer in range(self.args.n_outer_steps):
            self.log(f"\n{'#'*60}")
            self.log(f"OUTER ITERATION {outer + 1}/{self.args.n_outer_steps}")
            self.log(f"{'#'*60}")

            # 1. Select active Training Skill
            # Update per-type stats for compound conditions (e.g., weakest_type < 10%)
            self.state.update_type_stats_from_file(self._get_task_stats_path())
            active_ts = self.state.select_training_skill(training_skills)
            self.log(f"Active training skill: {active_ts.get('title', '?')}")
            self.log(f"  Formula: {active_ts.get('reward_formula', 'N/A')}")

            # 2. Inner loop: veRL GRPO
            ckpt = self.run_verl_grpo(
                outer_step=outer,
                training_skill=active_ts,
                n_epochs=self.args.n_inner_epochs,
            )

            # Save task stats from veRL output for next iteration's data selection
            verl_log = os.path.join(self.output_dir, f"verl_outer{outer}.log")
            if os.path.exists(verl_log):
                self._save_task_stats(verl_log)

            # 3. Dev eval (before evolution so trajectories feed diagnostics)
            if self.args.dev_eval and not self.args.dry_run:
                vllm_ok = self._start_vllm_server()
                self.run_dev_eval(outer)
                self._stop_vllm_server()

            # 4. Outer loop: skill evolution
            #    Uses dev_eval trajectories for diagnostics
            #    Training skills are unlocked every K outer steps
            K = self.args.training_skill_update_k
            training_locked = (outer % K != 0) or (outer == 0)
            self.log(f"  Training skill locked: {training_locked} "
                     f"(K={K}, step={outer})")

            if not self.args.skip_evolution and not self.args.dry_run:
                # Start vLLM server for proxy eval (uses GPUs freed by veRL)
                vllm_ok = self._start_vllm_server()
                if not vllm_ok:
                    self.log("  WARNING: vLLM server failed, proxy eval will use heuristic fallback")

            if not self.args.skip_evolution:
                training_skills = self.run_outer_evolution(
                    outer, training_skills, training_locked)

            if not self.args.skip_evolution and not self.args.dry_run:
                # Stop vLLM server to free GPU for next inner loop
                self._stop_vllm_server()

            # 5. Phase transition check + Phase 2 ratio ramp
            if self._phase == 1:
                if self._check_phase_transition():
                    self._execute_phase_transition()
            elif self._phase == 2:
                # Gradually increase model candidate ratio: 0.33 → 0.67 → 1.0
                self._model_candidate_ratio = min(1.0, self._model_candidate_ratio + 0.33)
                self.log(f"  Phase 2 model ratio: {self._model_candidate_ratio:.0%}")

            # 6. Save state
            self.state.outer_step = outer + 1
            self.state.history.append({
                "outer_step": outer,
                "success_rate": self.state.success_rate,
                "skills": self.current_skills,
                "model": self.current_model_path,
                "training_skill": active_ts.get("title", "?"),
                "phase": self._phase,
            })
            self.state.save()

        self.log(f"\n{'='*60}")
        self.log(f"Pipeline complete! (Phase {self._phase})")
        self.log(f"  Total epochs: {self.state.total_epochs}")
        self.log(f"  Total episodes: {self.state.total_episodes}")
        self.log(f"  Final success rate: {self.state.success_rate:.1%}")
        self.log(f"  Final model: {self.current_model_path}")
        self.log(f"  Final skills: {self.current_skills}")
        self.log(f"  Evolution rounds: {self._evolution_data_count}")
        self.state.save()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATS + Sphere Training Pipeline (veRL GRPO)")

    # Mode
    parser.add_argument("--mode", type=str, default="grpo",
                        choices=["grpo", "dry_run"],
                        help="Training mode")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Use debug GRPO script (minimal batch)")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick GRPO script (run_ats_grpo_quick.sh, tested params)")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank (0=full finetune, >0=LoRA for memory-constrained hardware)")

    # Skills
    parser.add_argument("--skills_path", type=str,
                        default="data/skills/appworld_skills_ats.json")
    parser.add_argument("--training_skills", type=str,
                        default="data/skills/training_skills.json")

    # Loop configuration
    parser.add_argument("--n_outer_steps", type=int, default=3,
                        help="Number of outer loop iterations")
    parser.add_argument("--n_inner_epochs", type=int, default=10,
                        help="veRL GRPO epochs per outer iteration")
    parser.add_argument("--tasks_per_epoch", type=int, default=32,
                        help="Train batch size per epoch")
    parser.add_argument("--val_tasks", type=int, default=16,
                        help="Validation batch size")
    parser.add_argument("--G", type=int, default=4,
                        help="GRPO group size (trajectories per task)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Max env steps per episode")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--test_freq", type=int, default=5,
                        help="Run validation every N epochs")

    # Evolution
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Candidates per evolution")
    parser.add_argument("--proxy_n_tasks", type=int, default=10,
                        help="Tasks for proxy eval per candidate")
    parser.add_argument("--training_skill_update_k", type=int, default=3,
                        help="Training skills can evolve every K outer steps")
    parser.add_argument("--skip_evolution", action="store_true",
                        help="Skip outer loop evolution (GRPO only)")
    parser.add_argument("--verifier_model", type=str, default="gpt-5.4")

    # Hardware
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--param_offload", action="store_true", default=False)
    parser.add_argument("--optimizer_offload", action="store_true", default=False)
    parser.add_argument("--free_cache_engine", action="store_true", default=False)
    parser.add_argument("--gpu_mem_util", type=float, default=0.4)
    parser.add_argument("--log_prob_micro_bs", type=int, default=2)

    # Model
    parser.add_argument("--model", type=str,
                        default="results/ats_sft_warmup/checkpoints/warmup_epoch3_hf")
    parser.add_argument("--backend", type=str, default="vllm")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking in dev eval (off by default: GRPO can't limit thinking_budget)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="results/ats_pipeline/")

    # Dev eval
    parser.add_argument("--dev_eval", action="store_true",
                        help="Run held-out dev eval after each outer iteration")

    # Phase 2
    parser.add_argument("--phase2_tgc_threshold", type=float, default=0.15)
    parser.add_argument("--phase2_data_threshold", type=int, default=25)

    # Timeouts
    parser.add_argument("--verl_timeout", type=int, default=43200,
                        help="veRL GRPO timeout in seconds (default 12h)")

    args = parser.parse_args()
    if args.mode == "dry_run":
        args.dry_run = True

    pipeline = ATSPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
