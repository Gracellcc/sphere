#!/bin/bash
set -x

# ATS+Sphere GRPO — DEBUG版本
#
# 最小化参数，快速验证完整pipeline：
#   - 4个train task × 2 group = 8条trajectory
#   - 2个val task
#   - 10步max (快速episode)
#   - 2个epoch (验证训练循环)
#   - 纯outcome reward (不调verifier API，省钱)
#   - Console日志 (不需要wandb)
#
# 预计运行时间: ~10-15分钟 (取决于AppWorld速度)
#
# Usage:
#   AZURE_OPENAI_API_KEY=xxx bash scripts/verl/run_ats_grpo_debug.sh
#   N_GPUS=2 bash scripts/verl/run_ats_grpo_debug.sh
#
# 验证checklist:
#   ✓ Patch applied (reward_manager='ats', env='appworld_ats')
#   ✓ Dataset loads tasks from appworld://train
#   ✓ VecEnv resets with task IDs from dataset
#   ✓ Env manager builds full template with history
#   ✓ Sphere retrieves skills per step
#   ✓ Code extraction (<code> or ```python)
#   ✓ Reward manager reads episode_rewards/episode_lengths
#   ✓ GRPO advantage computed across group
#   ✓ Actor loss computed and gradient flows
#   ✓ Validation success_rate logged

ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATASET_FILE="$ROOT_DIR/scripts/verl/dataset_ats.py"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export APPWORLD_ROOT=${APPWORLD_ROOT:-/home/yiyangai/srpo}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Apply veRL registry patch (idempotent) ────────────────────────────────
python3 "$ROOT_DIR/scripts/verl/patch_verl_registry.py"

# ── DEBUG parameters (最小化) ─────────────────────────────────────────────
model_path=${MODEL_PATH:-results/ats_sft_warmup/checkpoints/warmup_epoch3_hf}

# 数据: 极小batch
train_data_size=4                       # 4个不同task
val_data_size=2                         # 2个val task
group_size=2                            # 2条trajectory/task (GRPO最小)
max_steps=10                            # 10步max (快速)
max_prompt_length=4096
max_response_length=1024

# Reward: 纯outcome (不调verifier)
outcome_weight=1.0
supervision_weight=0.0                  # 关闭verifier → 省API钱
efficiency_weight=0.1                   # 轻微效率奖励看看是否生效

# Skills
skills_path=${SKILLS_PATH:-$ROOT_DIR/data/skills/appworld_skills_ats.json}

# Azure (verifier关了，但key仍需要因为reward manager init检查)
azure_endpoint=${AZURE_OPENAI_ENDPOINT:-https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com}
azure_api_key=${AZURE_OPENAI_API_KEY:-dummy_for_debug}

# Hardware (最小化)
n_gpus=${N_GPUS:-4}
tp_size=${TP_SIZE:-$n_gpus}

echo "============================================"
echo "  ATS GRPO DEBUG RUN"
echo "  Model: $model_path"
echo "  Train: ${train_data_size} tasks × ${group_size} group = $((train_data_size * group_size)) trajectories"
echo "  Val: ${val_data_size} tasks"
echo "  Max steps: ${max_steps}"
echo "  GPUs: ${n_gpus}, TP: ${tp_size}"
echo "  Reward: outcome=${outcome_weight}, supervision=${supervision_weight}, efficiency=${efficiency_weight}"
echo "  Skills: ${skills_path}"
echo "============================================"

# ── Launch veRL GRPO (debug config) ───────────────────────────────────────
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
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.custom_cls.path="file://$DATASET_FILE" \
    data.custom_cls.name=ATSDataset \
    +data.ats.skills_path=$skills_path \
    +data.ats.data_selection=balanced \
    +data.ats.task_stats_path="" \
    +data.ats.max_samples=$train_data_size \
    \
    actor_rollout_ref.model.path=$model_path \
    critic.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_data_size * group_size)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward_model.reward_manager=ats \
    +reward_model.reward_kwargs.outcome_weight=$outcome_weight \
    +reward_model.reward_kwargs.supervision_weight=$supervision_weight \
    +reward_model.reward_kwargs.efficiency_weight=$efficiency_weight \
    +reward_model.reward_kwargs.sgc_gate_tau=0.6 \
    +reward_model.reward_kwargs.verifier_model=gpt-5.4 \
    +reward_model.reward_kwargs.azure_endpoint=$azure_endpoint \
    +reward_model.reward_kwargs.azure_api_key=$azure_api_key \
    +reward_model.reward_kwargs.skills_path=$skills_path \
    +reward_model.reward_kwargs.max_steps=$max_steps \
    \
    algorithm.use_kl_in_reward=False \
    \
    env.env_name=appworld_ats \
    env.seed=42 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    +env.appworld_ats.max_steps=$max_steps \
    +env.appworld_ats.experiment_name=ats_grpo_debug \
    \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='ats_sphere_debug' \
    trainer.experiment_name='ats_grpo_debug' \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    "$@"

echo ""
echo "============================================"
echo "  DEBUG RUN COMPLETE"
echo "  Check logs for:"
echo "    - [ATS Dataset] task loading"
echo "    - [ATS EnvMgr] Sphere loaded"
echo "    - [ATS VecEnv] Reset/Step logs"
echo "    - [ATS Reward] episode scores"
echo "    - success_rate metrics"
echo "============================================"
