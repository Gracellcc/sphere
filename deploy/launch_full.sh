#!/bin/bash
# ATS+Sphere GRPO — Full Production (A100 8x80GB)
#
# Full-scale pipeline: GRPO training + skill evolution + phase transition
#
# Config:
#   - 32 tasks × 4 group = 128 trajectories/epoch
#   - 10 inner epochs × 5 outer steps = 50 total epochs
#   - save_freq=5, test_freq=5
#   - Full finetune (no LoRA), TP=1
#   - Dev eval every outer step
#   - Training skill evolution every 3 outer steps (K=3)
#
# Expected: ~6-12 hours (depends on AppWorld + verifier API speed)
#
# Usage:
#   AZURE_OPENAI_API_KEY=xxx bash deploy/launch_full.sh

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
    --n_outer_steps 5 \
    --n_inner_epochs 10 \
    --tasks_per_epoch 32 \
    --val_tasks 16 \
    --G 4 \
    --max_steps 30 \
    --save_freq 5 \
    --test_freq 5 \
    --n_gpus 8 \
    --tp_size 1 \
    --gpu_mem_util 0.6 \
    --log_prob_micro_bs 16 \
    --dev_eval \
    --training_skill_update_k 3 \
    --output_dir results/ats_pipeline_full/ \
    --verl_timeout 43200 \
    "$@"
