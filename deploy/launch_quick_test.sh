#!/bin/bash
# ATS+Sphere GRPO — Quick Test (A100 8x80GB)
#
# Small-scale but COMPLETE pipeline: inner GRPO + outer evolution + dev eval
# Purpose: Verify entire pipeline works on new hardware before full run
#
# Config:
#   - 16 tasks × 4 group = 64 trajectories/epoch
#   - 2 inner epochs × 2 outer steps = 4 total epochs
#   - save_freq=1, test_freq=1 (see every epoch)
#   - Full finetune (no LoRA), TP=1
#   - Checkpoint: FSDP + HF model (for vLLM loading)
#
# Expected: ~30-60 min
#
# Usage:
#   AZURE_OPENAI_API_KEY=xxx bash deploy/launch_quick_test.sh

set -e

# Verify API key
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "ERROR: AZURE_OPENAI_API_KEY must be set"
    exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

python scripts/ats_pipeline.py \
    --mode grpo \
    --model results/ats_sft_warmup/checkpoints/warmup_epoch3_hf \
    --skills_path data/skills/appworld_skills_ats.json \
    --n_outer_steps 2 \
    --n_inner_epochs 2 \
    --tasks_per_epoch 16 \
    --val_tasks 8 \
    --G 4 \
    --max_steps 20 \
    --save_freq 1 \
    --test_freq 1 \
    --n_gpus 8 \
    --tp_size 1 \
    --gpu_mem_util 0.6 \
    --log_prob_micro_bs 16 \
    --dev_eval \
    --training_skill_update_k 1 \
    --output_dir results/ats_pipeline_quick_test/ \
    --verl_timeout 7200 \
    "$@"
