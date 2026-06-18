#!/bin/bash
# ENFLAME GCU single-chip FL training example (verl-FL + Migration).
#
# Prerequisites:
#   - pip install migration  (ENFLAME GCU runtime patches)
#   - pip install verl-FL      (PlatformENFLAME builtin)
#   - torch_gcu, ECCL, FlagGems, TransformerEngine-FL, vllm-plugin-FL
#
# Startup order: Migration patches apply on import before verl initializes PlatformENFLAME.

set -x

# ============ Migration bootstrap (must run before verl) ============
export ENFLAME_ENABLE_AUTO_MIGRATION=1
export PYTHONPATH="/home/xijun.gong/icode/Megatron-LM-FL"
export PYTHONPATH="${PYTHONPATH:-}"
export RAY_DEDUP_LOGS=0
python3 -c "import verl; from verl.plugin.platform import get_platform; assert get_platform().device_name in ('enflame', 'cpu', 'cuda'); print(get_platform().device_name)"

# ============ ENFLAME Platform ============
export  ="${VERL_PLATFORM:-enflame}"
export TOPS_VISIBLE_DEVICES="${TOPS_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
# ============ FL / Communication (single-chip ECCL homogenous) ============
# export VERL_ENGINE_DEVICE=flagos
# export TE_FL_PREFER=flagos
export TE_FL_STRICT=0
# export USE_FLAGGEMS=true
export USE_FLAGGEMS=false
export VLLM_FL_OOT_ENABLED=1
export TRAIN_FILES="/home/xijun.gong/icode/gsm8k/train.parquet"
export VAL_FILES="/home/xijun.gong/icode/gsm8k/test.parquet"
export MODEL_PATH="/home/xijun.gong/Qwen3-0.6B"
export TEFL_LOG_LEVEL=DEBUG 
export TE_FL_SKIP_CUDA=1 
export TE_FL_PREFER=vendor 
export NVTE_DEBUG=1 
export NVTE_DEBUG_LEVEL=2 
export ENFLAME_MIGRATION_CACHE_DIR=./mycache  
export ENFLAME_MIGRATION_DUMP_DIR=./migration_debug 
export ENFLAME_MIGRATION_LOG_LEVEL=INFO
# Default: ECCL for ENFLAME homogenous cluster. Set USE_FLAGCX=1 for FlagCX instead.
export USE_FLAGCX="${USE_FLAGCX:-0}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +ray_kwargs.ray_init.runtime_env.env_vars.ENFLAME_ENABLE_AUTO_MIGRATION=\'1\' \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\'1\' \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=\'1\' \
    +ray_kwargs.ray_init.runtime_env.env_vars.ENFLAME_TE_KERNEL_TRITON_BACKEND=\'fused_rope_fwd,fused_rope_bwd\' \
    +ray_kwargs.ray_init.runtime_env.env_vars.TOPS_VISIBLE_DEVICES=\'0,1,2,3,4,5,6,7\' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\'/home/xijun.gong/icode/Megatron-LM-FL\' \
    data.train_files="${TRAIN_FILES:-./train.parquet}" \
    data.val_files="${VAL_FILES:-./test.parquet}" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH:-/path/to/Qwen3-0.6B}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_enflame_fl' \
    trainer.experiment_name='qwen3_0.6b_enflame_fl' \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.ray_wait_register_center_timeout=60 \
    trainer.use_legacy_worker_impl='disable' \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.total_epochs=15 \
    "$@"