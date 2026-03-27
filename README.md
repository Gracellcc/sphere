# ATS+Sphere: Adaptive Training Skills with Spherical Skill Injection

A two-loop reinforcement learning pipeline for training LLM agents on AppWorld. The inner loop (GRPO) trains the agent to solve tasks; the outer loop evolves the skill library that guides the agent. Sphere provides per-step geometric skill retrieval and injection at inference time — no extra training required.

```
J(theta) = J_inner(theta) + lambda_outer * J_outer(theta)
```

## Overview

### What Are Skills?

Two types of skills drive the pipeline:

**Behavioral Skills** (44 in the default library) teach the agent *what to do*:
- **Guidance**: concrete instructions injected into each step's prompt (e.g., "Call `apis.supervisor.show_account_passwords()` before accessing any app")
- **Scoring**: evaluation criteria fed to a verifier LLM for supervision reward
- **When to Use**: conditions for retrieval filtering
- **Category**: app-specific (`spotify`, `venmo`, `multi_app`) or universal (`general`, `common_mistakes`)
- **Embedding**: 1024d vector on a unit sphere (Qwen3-Embedding-0.6B, L2-normalized)

**Training Skills** (3 by default) control *how to train*:

| Training Skill | Activates When | Outcome Weight | Supervision Weight | Efficiency Weight | Data Selection |
|---|---|---|---|---|---|
| Early Stage Balanced | SR < 20% | 1.0 | 0.3 | 0.0 | balanced (exclude SR>80%) |
| Weakness Focused | SR > 20% AND weakest < 10% | 0.7 | 1.0 | 0.0 | 70% worst types + 30% random |
| Efficiency Push | SR > 40% | 0.5 | 1.0 | 0.3 | SR>30% but steps>20 |

### Architecture

```
ats_pipeline.py (orchestrator)
  for each outer_step:

    Phase 1 (API-driven) or Phase 2 (model-driven):

    1. Select Training Skill
       -> reward weights + data selection policy based on current SR

    2. Inner Loop: veRL GRPO (multiple epochs)
       -> FSDP training + vLLM rollout
       -> AppWorld multi-turn env + Sphere per-step injection
       -> Reward = outcome * w1 + supervision * w2 + efficiency * w3

    3. Dev Eval
       -> start vLLM -> evaluate held-out tasks -> stop vLLM

    4. Outer Loop: Skill Evolution
       Phase 1: GPT-5.4 generates G candidates -> proxy eval -> best updates skill bank
       Phase 2: Model generates G candidates (with log probs)
                -> proxy eval -> GRPO update on evolution capability
                -> best updates skill bank

    5. Phase Transition Check
       Phase 1 -> Phase 2 when: evolution_data >= threshold AND dev_TGC >= threshold
       Transition: SFT on accumulated (diagnostics, candidate, reward) triplets
       Then model_candidate_ratio ramps: 33% -> 67% -> 100%
```

---

## Sphere Injection Pipeline (Per-Step)

Every step before the LLM generates an action:

```
1. Encode        query = task_desc + last_action + observation
                 -> Qwen3-Embedding-0.6B -> 1024d -> L2 normalize -> point on unit sphere

2. Intent        drift = arccos(prev_intent . query_vec)
   Update        alpha = sigmoid(sensitivity * (drift/drift_typical - 1))  in [0.15, 0.6]
                 intent = slerp(prev_intent, query_vec, alpha)

3. SGC           coherence = displacement / path_length
   Confidence    stability = sigmoid(-4 * (recent_drift_norm - 1))
                 SGC = 0.55 * coherence + 0.45 * stability

4. Injection     isolation = d_nearest_skill / d_typical
   Decision      coverage_factor = sigmoid(-4 * (isolation - 1.0))
                 gamma = 1.5 if stuck, 1.3 if confused, 1.0 otherwise (never < 1.0)

5. Retrieval     top-K cosine -> category filter -> adaptive threshold filter
                 -> skill rotation penalty (0.15 on recently used)
                 -> greedy complementary selection: score = relevance * (0.5 + 0.5*sin(theta))
                 -> redundancy check (cos > 0.85 -> skip)
                 -> spherical excess quality gate (Eriksson's formula on 3D projection)

6. Weights       w_i = gamma * base_strength * coverage_factor * exp(-d_norm_i^2 / sigma^2)
                 base_strength = SGC * 0.8 + 0.2
                 filter: w_i < 0.05 -> discard

7. Combine       combined = multi_slerp(selected_skills, weights)
   & Format      re-rank by geodesic distance to combined vector (Frechet mean)
                 format as numbered skill list with confidence scores
                 inject into prompt's {skill_section}
```

### Self-Calibrating Thresholds

All geometric thresholds are computed from the skill bank itself — no per-environment tuning:
- `adaptive_threshold` = p50 of all pairwise skill cosine similarities
- `d_typical` = median pairwise geodesic distance
- `min_excess` = p25 of sampled spherical excess triples

### Why Sphere Geometry (Not Flat Cosine)?

- **slerp**: interpolation stays on the unit sphere; linear interpolation drifts off
- **geodesic distance**: `arccos(dot)` captures angular separation, not magnitude
- **spherical excess**: measures the "information spread" of a skill triple — flat geometry has no equivalent
- **sin(theta) complementarity**: more sensitive to small angular differences than `1-cos(theta)`
- **coverage gate**: isolation as `d_nn / d_typical` is scale-invariant across different skill banks

---

## Reward System

```
reward = outcome_weight * outcome + supervision_weight * supervision + efficiency_weight * efficiency
```

**Outcome**: AppWorld task completion (0 or 1).

**Supervision**: Per-skill scoring by a frozen verifier LLM (GPT-5.4).
- SGC Gate: if `outcome == 1 AND SGC > 0.6`, skip verifier API call, use SGC as supervision score (saves 40-60% API cost)
- Otherwise: each active behavioral skill's Scoring criteria + trajectory -> verifier -> 0-1 score -> average
- Programmatic scorers for rules like `count_ratio`, `pattern_before`, `order_check`
- LLM verifier for natural-language scoring criteria
- Supervision cache: `hash(trajectory_text)` -> reuse scores for identical trajectories

**Efficiency**: `max(0, 1 - steps_used / max_steps)`. Only active when SR > 40%.

---

## Outer Loop: Skill Evolution

### Phase 1 (API-Driven)

Each outer step:
1. Collect diagnostics: per-type SR, error categories, sphere spatial info (redundancy, isolation)
2. GPT-5.4 reads diagnostics + current skill bank -> generates G=3 candidate modifications
3. Each candidate: temporarily apply changes -> proxy eval on ~10 weak tasks -> reward
4. Best candidate -> update skill bank, re-encode modified skills into sphere
5. Accumulate (diagnostics, candidate, reward) triplets for Phase 2 transition

Training skill is locked for K-1 out of every K outer steps (default K=3).

### Phase 2 (Model-Driven GRPO)

Triggered when `evolution_data >= phase2_data_threshold AND dev_TGC >= 0.15`.

**Transition**: SFT on accumulated triplets (LoRA r=64, 2 epochs) teaches the model to generate skill modifications.

Each outer step:
1. Collect diagnostics
2. **Model generates** G candidates with per-token log probs (OuterGRPO)
3. Proxy eval each candidate -> rewards
4. **GRPO update**: `J_outer = -E[advantage * log pi(candidate | diagnostics)]`
   - advantage = reward_i - mean(rewards) (group-relative, no std normalization)
   - clipped surrogate with epsilon=0.2
   - scaled by `lambda_outer` (default 0.1)
5. Save updated model, best candidate -> update skill bank

`model_candidate_ratio` ramps from 33% -> 67% -> 100% across outer steps.

---

## Setup

### Prerequisites

```bash
# 1. Clone and install this repo
git clone https://github.com/Gracellcc/sphere.git
cd sphere
pip install -e .

# 2. Install SelfSkill (veRL fork) and apply patch
git clone https://github.com/plzdoo/SelfSkill.git
cd SelfSkill
git apply ../sphere/deploy/selfskill.patch
pip install -e .
cd ..

# 3. Install AppWorld
pip install appworld
appworld install

# 4. Create AppWorld conda environment (required for pydantic v1 compatibility)
conda create -n appworld_env python=3.10 -y
conda activate appworld_env
pip install appworld
conda deactivate

# 5. Download warmup checkpoint (or use Qwen3-8B base)
huggingface-cli download Gracecc/qwen3-8b-sphere-epoch3 --local-dir warmup_epoch3_hf

# 6. Install flash attention (required by Qwen3)
pip install flash-attn --no-build-isolation
```

### Environment Variables

```bash
# Required
export APPWORLD_ROOT=/path/to/appworld          # contains data/datasets/{train,dev,test_normal,test_challenge}.txt
export SELFSKILL_ROOT=/path/to/SelfSkill        # veRL fork with patch applied
export AZURE_OPENAI_API_KEY=your_key_here       # for verifier + skill evolution
export AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com

# Optional
export WANDB_API_KEY=your_wandb_key             # for training logging
export APPWORLD_CONDA_ENV=appworld_env          # default: appworld_env
```

See `.env.example` for the full list.

### Verify Setup

```bash
# Check AppWorld data
ls $APPWORLD_ROOT/data/datasets/train.txt       # should have 89 task IDs

# Check skills
python -c "import json; print(len(json.load(open('data/skills/appworld_skills_ats.json'))))"  # 44

# Check model
ls warmup_epoch3_hf/config.json                 # should exist

# Dry run
AZURE_OPENAI_API_KEY=test AZURE_OPENAI_ENDPOINT=https://test.com \
  python scripts/ats_pipeline.py --dry_run --model warmup_epoch3_hf \
  --skills_path data/skills/appworld_skills_ats.json --n_outer_steps 1
```

---

## Launch

```bash
# Quick test (~30-60min): 16 tasks, 2 epochs, 2 outer steps
AZURE_OPENAI_API_KEY=xxx bash deploy/launch_quick_test.sh

# Full production (~6-12h): 32 tasks, 10 epochs, 5 outer steps
AZURE_OPENAI_API_KEY=xxx bash deploy/launch_full.sh
```

| Config | Quick Test | Full |
|--------|-----------|------|
| Tasks/epoch | 16 | 32 |
| Group size (G) | 4 | 4 |
| Inner epochs/outer step | 2 | 10 |
| Outer steps | 2 | 5 |
| save/test_freq | 1 | 5 |
| Max steps/episode | 20 | 30 |
| Training skill update K | 1 | 3 |
| Phase 2 data threshold | 25 (won't trigger) | 3 |

Override any parameter:
```bash
bash deploy/launch_quick_test.sh --n_gpus 4 --tp_size 2
bash deploy/launch_full.sh --skip_evolution       # inner GRPO only, no evolution
bash deploy/launch_full.sh --lambda_outer 0.2     # stronger outer GRPO
bash deploy/launch_quick_test.sh --dry_run        # simulate without execution
```

---

## Monitoring

```bash
# Pipeline progress
tail -f results/ats_pipeline_*/pipeline.log

# veRL inner loop
tail -f results/ats_pipeline_*/verl_outer0.log

# wandb: project = ats_sphere_appworld
```

Key log tags: `[ATS Dataset]`, `[ATS EnvMgr]`, `[ATS VecEnv]`, `[ATS Reward]`, `success_rate`

---

## Skill Bank

44 behavioral skills in `data/skills/appworld_skills_ats.json`:

| Category | Count | Examples |
|----------|-------|---------|
| general | 13 | Stop doc-looping, verify side effects, paginate fully |
| spotify | 7 | Correct library endpoints, album as ID indexes |
| venmo | 6 | Session management, transaction direction filtering |
| multi_app | 6 | Source/destination mapping, backup-before-delete |
| file_system | 6 | List before write, verify after write |
| execution_bootstrap | 1 | Take a live first step within 2 actions |

Category filtering ensures task-irrelevant skills are never injected (e.g., a Gmail-only task won't see Spotify skills). Universal categories (`general`, `common_mistakes`, `multi_app`) are always included.

---

## File Structure

```
scripts/
  ats_pipeline.py                # Main orchestrator (Phase 1 + Phase 2)
  evolve_skills.py               # Evolution prompt + EVOLUTION_SYSTEM_PROMPT
  train_transition_sft.py        # Phase 2 transition SFT (LoRA)
  verl/
    run_ats_grpo.sh              # veRL GRPO launch script
    env_appworld_ats.py          # AppWorld env + Sphere integration + prompt templates
    dataset_ats.py               # Task loading + data selection policies
    reward_manager_ats.py        # Outcome + supervision (verifier/SGC gate) + efficiency
    ats_outer_loop.py            # Diagnostics + candidate generation + proxy eval
    outer_grpo.py                # Phase 2 outer GRPO (generate with log probs + update)
    patch_verl_registry.py       # Patch SelfSkill dispatch tables for ATS

skill_sphere/
  geometry/
    sphere.py                    # geodesic_distance, slerp, multi_slerp, L2 normalize
    excess.py                    # spherical_excess (Eriksson), combination_diversity
    tangent.py                   # Tangent space operations
    voronoi.py                   # Spherical Voronoi for coverage analysis
  skill_bank/
    encoder.py                   # Qwen3-Embedding-0.6B (1024d, sentence_transformers)
    skill_sphere.py              # Skill library: load JSON -> encode -> store on sphere
    retrieval.py                 # Calibration + complementarity-aware retrieval
    combination.py               # Skill combination utilities
  injection/
    intent_tracker.py            # Adaptive momentum intent tracking (slerp, sigmoid alpha)
    confidence.py                # SGC = 0.55*coherence + 0.45*stability
    dynamic_inject.py            # Full injection controller (isolation, gamma, weights)
  agent/
    appworld_agent.py            # Agent wrapper for evaluation
    llm_client.py                # OpenAI-compatible LLM client
  env/
    appworld_wrapper.py          # AppWorld subprocess wrapper (pydantic v1/v2 isolation)
    appworld_server.py           # AppWorld subprocess server (JSON-line protocol)

data/skills/
  appworld_skills_ats.json       # 44 behavioral skills
  training_skills.json           # 3 training skills

deploy/
  launch_quick_test.sh           # Quick test config (16 tasks, 2 epochs)
  launch_full.sh                 # Production config (32 tasks, 10 epochs)
  selfskill.patch                # Patch for SelfSkill (veRL fork)
  skill.md                       # Deployment guide + troubleshooting
```

---

## Hardware

Tested on:
- **A100 8x80GB**: Full finetune (no LoRA), TP=1, no offload. Recommended.
- **A6000 4x48GB**: Works with `--param_offload --optimizer_offload` or `--lora_rank 16`.

The pipeline auto-detects GPU count and adjusts mini-batch size:
```
is_large_gpu = (n_gpus >= 4) and (lora_rank == 0)
-> ppo_mini_batch_size = tasks_per_epoch * G / 4
```

---

## Checkpoint Format

veRL saves checkpoints as:
```
global_step_N/
  actor/                         # FSDP shards (for veRL resume)
  actor/huggingface/             # HF format (for vLLM loading + dev eval)
  optimizer/                     # Optimizer states
```

The pipeline auto-detects `actor/huggingface/` with `config.json` and uses the HF path for vLLM/evaluation, the raw path for veRL resume.

---

## Known Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Qwen3 thinking eats all tokens | GRPO can't limit `thinking_budget` | `enable_thinking=False` everywhere |
| vLLM can't load FSDP checkpoint | Wrong format | `hf_model` in checkpoint contents, auto path routing |
| Doc-looping (72% of failures) | 8B model loops on `api_docs.show_api_doc()` | Behavioral skills #1 + #44 specifically target this |
| Supervisor info empty in prompt | `appworld_env` conda env missing | Create env: `conda create -n appworld_env python=3.10` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| vLLM OOM | `--gpu_mem_util 0.4` or `--tp_size 2` |
| FSDP OOM | `--param_offload --optimizer_offload` or `--lora_rank 16` |
| `ModuleNotFoundError: flash_attn` | `pip install flash-attn --no-build-isolation` |
| `APPWORLD_ROOT must be set` | `export APPWORLD_ROOT=/path/to/appworld` |
| `SELFSKILL_ROOT must be set` | `export SELFSKILL_ROOT=/path/to/SelfSkill` |
| `AZURE_OPENAI_ENDPOINT must be set` | `export AZURE_OPENAI_ENDPOINT=https://...` |
| AppWorld reset fails | Check `$APPWORLD_ROOT/data/datasets/` has `train.txt` etc. |
| All episodes fail in epoch 1 | Check `appworld_env` conda env exists and has `appworld` installed |
| No HF checkpoint saved | Verify `hf_model` in veRL checkpoint contents |
| Phase 2 never triggers | Lower `--phase2_data_threshold` (full.sh uses 3) |

---

## Config Flow

```
Pipeline args (--tasks_per_epoch 32)
  -> env vars (TRAIN_DATA_SIZE=32)
    -> run_ats_grpo.sh (train_data_size=${TRAIN_DATA_SIZE:-32})
      -> Hydra config (data.train_batch_size=$train_data_size)

Pipeline also appends Hydra overrides via "$@" which OVERRIDE shell-level values.
```

---

## Design Decisions

**Why behavioral skills as natural language, not executable code?**
Skills like "Stop doc-looping after one orienting check" teach the agent *how to think*, not *what code to write*. This generalizes across tasks better than hardcoded functions.

**Why per-step injection, not per-task?**
The agent's needs change within an episode: step 1 needs "get credentials", step 5 needs "verify side effects". Intent tracking on the sphere captures this drift.

**Why boost-only gamma (never < 1.0)?**
Suppressing injection when confidence is high risks removing useful guidance during critical steps. Boost-only ensures skills are always at least baseline strength, with extra push when the agent is stuck or confused.

**Why spherical excess as quality gate?**
Three skills that are nearly coplanar on the sphere provide redundant information. Spherical excess (the area of the spherical triangle) measures how much "new direction" a third skill adds. Below a threshold, it's not worth injecting.

**Why two phases?**
Phase 1 uses a strong API (GPT-5.4) to generate high-quality skill modifications while the model is still weak. Phase 2 transitions to model-driven evolution via GRPO, so the model learns to improve its own training — the key insight of ATS.
