# ATS+Sphere GRPO Deployment (A100 8x80GB)

## Setup

```bash
# 1. Dependencies: veRL, vLLM, appworld, torch, transformers
# 2. Set paths
export APPWORLD_ROOT=/path/to/appworld   # contains data/datasets/{train,dev,test_normal}.txt
export AZURE_OPENAI_API_KEY=<your-key>

# 3. Verify
ls $APPWORLD_ROOT/data/datasets/train.txt          # 89 tasks
ls results/ats_sft_warmup/checkpoints/warmup_epoch3_hf/config.json
python -c "import json; print(len(json.load(open('data/skills/appworld_skills_ats.json'))))"  # 47
```

## Launch

```bash
# Quick test first (~30-60min): 16 tasks, 2 epochs, 2 outer steps
AZURE_OPENAI_API_KEY=xxx bash deploy/launch_quick_test.sh

# Full production (~6-12h): 32 tasks, 10 epochs, 5 outer steps
AZURE_OPENAI_API_KEY=xxx bash deploy/launch_full.sh
```

| Config | Quick Test | Full |
|--------|-----------|------|
| Tasks/epoch | 16 | 32 |
| Group size (G) | 4 | 4 |
| Inner epochs | 2 | 10 |
| Outer steps | 2 | 5 |
| save/test_freq | 1 | 5 |
| Max steps/episode | 20 | 30 |

Override any parameter:
```bash
bash deploy/launch_quick_test.sh --n_gpus 4 --tp_size 2
bash deploy/launch_full.sh --skip_evolution   # GRPO only, no skill evolution
bash deploy/launch_quick_test.sh --dry_run    # simulate without execution
```

## Hardware (A100 8x80GB)

- Full finetune (no LoRA), TP=1, no offload
- GPU mem util: 0.6 for vLLM rollout
- Checkpoint: FSDP shards + HF model (auto-detected for vLLM loading)

## Config Flow

```
pipeline args (--tasks_per_epoch 32)
  -> env vars (TRAIN_DATA_SIZE=32)
    -> run_ats_grpo.sh (train_data_size=${TRAIN_DATA_SIZE:-32})
      -> Hydra config (data.train_batch_size=$train_data_size)

Pipeline also appends Hydra overrides via "$@" which OVERRIDE shell-level values.
```

## Pipeline Architecture

```
ats_pipeline.py (orchestrator)
  for each outer_step:
    1. Select Training Skill -> reward weights + data selection policy
    2. run_verl_grpo() -> FSDP train + vLLM rollout + AppWorld env + Sphere injection
    3. Dev eval (optional) -> start vLLM -> evaluate -> stop vLLM
    4. Skill evolution -> diagnostics -> GPT-5.4 candidates -> proxy eval
    5. Phase transition check (Phase 1: API-driven -> Phase 2: model-driven)
```

## Monitoring

```bash
tail -f results/ats_pipeline_*/pipeline.log     # pipeline progress
tail -f results/ats_pipeline_*/verl_outer0.log   # veRL inner loop
```

Key log tags: `[ATS Dataset]`, `[ATS EnvMgr]`, `[ATS VecEnv]`, `[ATS Reward]`, `success_rate`

## Known Issues (all fixed)

1. **Checkpoint format**: veRL FSDP shards can't be loaded by vLLM. Fix: `hf_model` added to checkpoint contents, pipeline auto-detects `actor/huggingface/` path.

2. **Qwen3 thinking mode**: Disabled everywhere (`enable_thinking=False`). GRPO can't limit `thinking_budget`, so thinking would eat all `max_response_length` and truncate code actions.

3. **Env reset crash**: `_process_batch` could fail assertion if no active step found. Fix: fallback `append(0.0)`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM | Reduce `--gpu_mem_util` to 0.4 |
| FSDP OOM | Add `--param_offload --optimizer_offload`, or use `--lora_rank 16` |
| AppWorld reset fails | Check `$APPWORLD_ROOT/data/datasets/` has split files |
| No HF checkpoint | Check veRL log for checkpoint saving; verify `hf_model` in contents |
