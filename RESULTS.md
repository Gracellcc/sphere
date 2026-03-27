# Skill Sphere — Evaluation Results (Historical)

> Model: Qwen2.5-7B-Instruct (untrained), vLLM backend
> Eval protocol: SkillRL-aligned (eval_in_distribution, max_history=2)
> Date: 2026-02-27
>
> **NOTE**: These results used skills distilled from incomplete memories (no BCC).
> Re-evaluation with properly generated skills is pending.

---

## 1. Main Results — Large-Scale Comparison

All runs use **Qwen2.5-7B-Instruct (untrained)** with identical eval protocol.

### ALFWorld (140 episodes, eval_in_distribution)

| Mode | Skills | Success Rate | N |
|------|--------|-------------|---|
| none (vanilla) | — | 26.4% (37/140) | 140 |
| skillrl_fair | GPT-5.2 distilled | 28.6% (40/140) | 140 |
| **sphere** | **GPT-5.2 distilled** | **32.9% (46/140)** | **140** |
| sphere | SkillRL original | 26.4% (37/140) | 140 |

> sphere > fair by **+4.3%** with same skills.

### Search QA (200-350 episodes, exact match)

| Mode | Skills | EM | N |
|------|--------|------|---|
| none (vanilla) | — | 4.5% (9/200) | 200 |
| skillrl_fair | GPT-5.2 distilled | 11.7% (41/350) | 350 |
| **sphere** | **GPT-5.2 distilled** | **12.6% (44/350)** | **350** |

> sphere > fair by **+0.9%** (marginal).

### WebShop (200 episodes, success rate / avg score)

| Mode | Skills | SR | Avg Score | N |
|------|--------|------|-----------|---|
| none (vanilla) | — | 7.5% | 0.368 | 200 |
| skillrl_fair | GPT-5.2 distilled | 2.5% | 0.163 | 200 |
| **sphere** | **GPT-5.2 distilled** | **11.0%** | **0.360** | **200** |

> sphere > fair by **+8.5% SR**. Fair actually hurts (2.5% < 7.5% none).

### ScienceWorld (149 episodes, avg score %)

| Mode | Skills | Avg Score % | N |
|------|--------|------------|---|
| none (vanilla) | — | -52.3% | 90 |
| skillrl_fair | GPT-5.2 distilled | -58.6% | 149 |
| **sphere** | **GPT-5.2 distilled** | **-35.2%** | **149** |

> sphere > fair by **+23.4%**. Largest improvement across all envs.

---

## 2. Summary: Sphere vs Fair (best config per env)

| Environment | none | skillrl_fair | sphere | sphere vs fair | sphere vs none |
|-------------|------|-------------|--------|----------------|----------------|
| ALFWorld | 26.4% | 28.6% | **32.9%** | **+4.3%** | **+6.5%** |
| Search | 4.5% | 11.7% | **12.6%** | **+0.9%** | **+8.1%** |
| WebShop | 7.5% | 2.5% | **11.0%** | **+8.5%** | **+3.5%** |
| ScienceWorld | -52.3% | -58.6% | **-35.2%** | **+23.4%** | **+17.1%** |

**Sphere > Fair in 4/4 environments** without any training.

---

## 3. Dev Iterations (30ep, same episode set, Qwen2.5-7B)

Tracked on the same 30 ALFWorld episodes (task dist: 9/5/4/5/5/2):

| Iteration | ALFWorld SR | Key Change |
|-----------|-----------|------------|
| none | 36.7% | No skills |
| skillrl_fair | 53.3% | Static template retrieval |
| sphere (basic) | **63.3%** | Basic sphere retrieval + dynamic inject |
| + adaptive threshold | 56.7% | Sphere-adaptive threshold |
| + all features | 50.0% | γ factor, alignment modulation, excess gate |
| + boost-only | 46.7% | Boost-only γ, skill rotation, removed suppressors |

> Note: 30ep has high variance. Large-scale (140-200ep) results in Section 1 are authoritative.

---

## 4. Bottleneck Analysis

### Why sphere improvements plateau with untrained Qwen-7B

1. **Confidence uncalibrated**: Mean confidence = 0.84 for both successes and failures.
   γ factor and rotation (conf < 0.4 threshold) almost never trigger.

2. **Exploration too weak**: Agent visits only 2-3 locations before looping.

3. **Doesn't follow injected skills**: Skills are injected but agent ignores them.

4. **These are training problems, not retrieval problems**:
   - SFT will teach task-solving patterns + skill following
   - GRPO will calibrate confidence + improve exploration
   - Sphere features (γ, rotation, drift) require calibrated confidence to activate

---

## 5. Reference: SkillRL Paper Numbers

| Method | ALFWorld | WebShop SR | WebShop Score |
|--------|----------|-----------|---------------|
| SkillRL (trained) | 89.9% | 72.7% | 85.2 |
| GRPO baseline | 77.6% | 66.1% | 79.3 |
| w/o Cold-Start SFT | 65.2% | 46.5% | — |
| Vanilla Qwen-7B | ~26% | ~7% | — |

Our untrained sphere (32.9% ALFWorld, 11.0% WebShop) is between vanilla and w/o SFT.
After training (Phase 5), target: sphere + trained model ≈ SkillRL (89.9% / 72.7%).

---

## 6. Results Directory Structure

```
results/
├── canonical/                    # Authoritative results
│   ├── phase3_140ep/             # Large-scale, early skills (140-200ep)
│   ├── qwen7b_30ep/             # Dev iterations (30ep same episodes)
│   └── trajectories/             # Step-by-step trajectory logs
└── archive/                      # Historical / debugging
    ├── gpt4o_30ep/               # GPT-4o experiments (old protocol)
    ├── phase0_alignment/         # Eval protocol alignment experiments
    └── intermediate/             # FAC diagnosis, synthesis, test runs
```
