#!/bin/bash
set -x

# ATS+Sphere GRPO — QUICK FULL版本
#
# 完整pipeline（Sphere注入 + Verifier评分 + SGC gate + GRPO），但数据量小跑得快。
# 能看到的效果：
#   - Sphere per-step skill注入是否工作
#   - Verifier supervision reward数值
#   - SGC gate是否正确跳过部分verifier调用
#   - GRPO reward趋势（是否epoch间有变化）
#   - Validation success_rate
#
# 配置：
#   - 16个train task × 4 group = 64条trajectory/epoch
#   - 8个val task
#   - 20步max (够完成大部分AppWorld任务)
#   - 5个epoch (看趋势)
#   - supervision_weight=0.3 (开启verifier，完整ATS reward)
#   - SGC gate tau=0.6 (高SGC+成功→跳过verifier)
#
# 预计运行时间: ~30-45分钟
# 预计verifier API调用: ~100-200次 (SGC gate省40-60%)
#
# Usage:
#   AZURE_OPENAI_API_KEY=xxx bash scripts/verl/run_ats_grpo_quick.sh

ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATASET_FILE="$ROOT_DIR/scripts/verl/dataset_ats.py"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export APPWORLD_ROOT=${APPWORLD_ROOT:-/home/yiyangai/srpo}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export MY_HOST_IP=${MY_HOST_IP:-127.0.0.1}

# ── Apply veRL registry patch (idempotent) ────────────────────────────────
python3 "$ROOT_DIR/scripts/verl/patch_verl_registry.py"

# ── Quick-full parameters ─────────────────────────────────────────────────
model_path=${MODEL_PATH:-results/ats_sft_warmup/checkpoints/warmup_epoch3_hf}

# 数据: 够看趋势但不太慢
train_data_size=16                      # 16个不同task
val_data_size=8                         # 8个val task
group_size=4                            # 4条trajectory/task (GRPO标准)
max_steps=20                            # 20步max (平衡速度和完整性)
max_prompt_length=5120
max_response_length=1024

# Reward: 完整ATS (开启verifier)
outcome_weight=1.0
supervision_weight=0.3                  # 开启verifier → 完整supervision信号
efficiency_weight=0.1                   # 轻微效率奖励
sgc_gate_tau=0.6

# Skills
skills_path=${SKILLS_PATH:-$ROOT_DIR/data/skills/appworld_skills_ats.json}

# Azure OpenAI (verifier)
azure_endpoint=${AZURE_OPENAI_ENDPOINT:-https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com}
azure_api_key=${AZURE_OPENAI_API_KEY:?AZURE_OPENAI_API_KEY must be set}
verifier_model=${VERIFIER_MODEL:-gpt-5.4}

# Hardware (4x A6000)
n_gpus=${N_GPUS:-4}
tp_size=${TP_SIZE:-2}                        # TP=2 (matches veRL official 8B config)

echo "============================================"
echo "  ATS GRPO QUICK-FULL RUN"
echo "  Model: $model_path"
echo "  Train: ${train_data_size} tasks × ${group_size} group = $((train_data_size * group_size)) trajectories/epoch"
echo "  Val: ${val_data_size} tasks"
echo "  Max steps: ${max_steps}, Epochs: 1"
echo "  GPUs: ${n_gpus}, TP: ${tp_size}"
echo "  Reward: outcome=${outcome_weight}, supervision=${supervision_weight}, efficiency=${efficiency_weight}"
echo "  SGC gate tau: ${sgc_gate_tau}"
echo "  Verifier: ${verifier_model}"
echo "  Skills: ${skills_path}"
echo "============================================"

# ── Launch veRL GRPO ──────────────────────────────────────────────────────
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
    +data.ats.task_stats_path="" \
    +data.ats.max_samples=$train_data_size \
    \
    actor_rollout_ref.model.path=$model_path \
    critic.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_data_size * group_size)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
    env.seed=42 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    +env.appworld_ats.max_steps=$max_steps \
    +env.appworld_ats.experiment_name=ats_grpo_quick \
    \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='ats_sphere_appworld' \
    trainer.experiment_name=${EXPERIMENT_NAME:-ats_grpo_quick_full} \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.val_before_train=True \
    ${DEFAULT_LOCAL_DIR:+trainer.default_local_dir=$DEFAULT_LOCAL_DIR} \
    "$@"

echo ""
echo "============================================"
echo "  QUICK-FULL RUN COMPLETE"
echo ""
echo "  关注的指标:"
echo "    - [ATS Reward] outcome/supervision/sgc/tgc数值"
echo "    - [ATS Reward] SGC gate skips (省了多少verifier调用)"
echo "    - success_rate 是否epoch间上升"
echo "    - wandb: ats_sphere_appworld/ats_grpo_quick_full"
echo "============================================"
