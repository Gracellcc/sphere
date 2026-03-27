#!/bin/bash
set -x

# ATS+Sphere GRPO Training on AppWorld via veRL
#
# Design doc: ATS_Sphere_design.md
# Inner loop: GRPO with Sphere per-step skill injection
# Outer loop: ats_outer_loop.py (run separately, updates skills between epochs)
#
# Hardware: 4x A6000 48GB
# Model: Qwen3-8B (or SFT checkpoint)
#
# IMPORTANT: Before running, apply the registration patch:
#   python scripts/verl/patch_verl_registry.py
# This adds 'ats' reward manager and 'appworld_ats' env to veRL's hardcoded dispatch.
#
# Usage:
#   bash scripts/verl/run_ats_grpo.sh [vllm|sglang] [extra verl args...]
#   MODEL_PATH=/path/to/sft/ckpt bash scripts/verl/run_ats_grpo.sh

ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATASET_FILE="$ROOT_DIR/scripts/verl/dataset_ats.py"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export APPWORLD_ROOT=${APPWORLD_ROOT:?APPWORLD_ROOT must be set}
# Make our modules importable
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Apply veRL registry patch (idempotent) ────────────────────────────────
python3 "$ROOT_DIR/scripts/verl/patch_verl_registry.py"

# ── Configurable parameters ──────────────────────────────────────────────
model_path=${MODEL_PATH:-results/ats_sft_warmup/checkpoints/warmup_epoch3_hf}
train_data_size=${TRAIN_DATA_SIZE:-32}
val_data_size=${VAL_DATA_SIZE:-16}
group_size=${GROUP_SIZE:-4}              # N trajectories per task (GRPO group)
max_steps=${MAX_STEPS:-30}               # Max env steps per episode
max_prompt_length=${MAX_PROMPT_LENGTH:-5120}
max_response_length=${MAX_RESPONSE_LENGTH:-3072}

# ATS reward weights (from active training skill, default=Early Stage Balanced)
outcome_weight=${OUTCOME_WEIGHT:-1.0}
supervision_weight=${SUPERVISION_WEIGHT:-0.3}
efficiency_weight=${EFFICIENCY_WEIGHT:-0.0}
sgc_gate_tau=${SGC_GATE_TAU:-0.6}

# Skills
skills_path=${SKILLS_PATH:-$ROOT_DIR/data/skills/appworld_skills_ats.json}
task_stats_path=${TASK_STATS_PATH:-}

# Azure OpenAI (verifier)
azure_endpoint=${AZURE_OPENAI_ENDPOINT:?AZURE_OPENAI_ENDPOINT must be set}
azure_api_key=${AZURE_OPENAI_API_KEY:?AZURE_OPENAI_API_KEY must be set}
verifier_model=${VERIFIER_MODEL:-gpt-5.4}

# Hardware (4x A6000)
n_gpus=${N_GPUS:-4}
tp_size=${TP_SIZE:-2}                            # tensor parallel for rollout vLLM
num_cpus_per_env_worker=0.1

# ── Launch veRL GRPO ─────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files='appworld://train+dev+test_challenge' \
    data.val_files='appworld://test_normal' \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.custom_cls.path="file://$DATASET_FILE" \
    data.custom_cls.name=ATSDataset \
    +data.ats.skills_path=$skills_path \
    +data.ats.data_selection=${DATA_SELECTION:-balanced} \
    +data.ats.task_stats_path="$task_stats_path" \
    +data.ats.max_samples=$train_data_size \
    \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.lora_rank=${LORA_RANK:-0} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-0} \
    critic.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-False} \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS:-16} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-False} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward_model.reward_manager=ats \
    +reward_model.reward_kwargs.outcome_weight=$outcome_weight \
    +reward_model.reward_kwargs.supervision_weight=$supervision_weight \
    +reward_model.reward_kwargs.efficiency_weight=$efficiency_weight \
    +reward_model.reward_kwargs.sgc_gate_tau=$sgc_gate_tau \
    +reward_model.reward_kwargs.verifier_model=$verifier_model \
    +reward_model.reward_kwargs.azure_endpoint=$azure_endpoint \
    +reward_model.reward_kwargs.azure_api_key=$azure_api_key \
    +reward_model.reward_kwargs.skills_path=$skills_path \
    +reward_model.reward_kwargs.max_steps=$max_steps \
    \
    algorithm.use_kl_in_reward=False \
    \
    env.env_name=appworld_ats \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.appworld_ats.max_steps=$max_steps \
    +env.appworld_ats.experiment_name=ats_grpo \
    \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='ats_sphere_appworld' \
    trainer.experiment_name=${EXPERIMENT_NAME:-ats_grpo_phase1} \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-5} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-30} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    ${DEFAULT_LOCAL_DIR:+trainer.default_local_dir=$DEFAULT_LOCAL_DIR} \
    "$@"
