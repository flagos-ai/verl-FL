# FL Multi-Chip Support Architecture Design

Last updated: 12/02/2026.

## Overview

This document describes the FL (FlagOS) multi-chip support architecture in verl, enabling training and inference on diverse hardware platforms through FlagGems, FlagCX, TransformerEngine-FL, and vllm-plugin-FL.

## Design Principles

1. **Environment Variable Driven+config file**: All FL configurations are controlled through environment variables or configuration files, with no parameter injection into existing verl APIs. This ensures minimal code intrusion and maximum compatibility.

2. **Phase Separation**: Training and Rollout phases have independent environment variable namespaces, allowing fine-grained control over each phase's operator and communication settings.

3. **Lightweight Integration**: FL engines extend existing engines (Megatron/FSDP) with minimal modifications, primarily adding environment validation and FlagGems initialization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           verl FL Multi-Chip Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         FLEnvManager                                    │   │
│  │   Unified environment variable management for Training & Rollout phases │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                          │                             │                       │
│            ┌─────────────▼─────────────┐  ┌────────────▼────────────┐          │
│            │     Training Phase        │  │     Rollout Phase       │          │
│            │                           │  │                         │          │
│            │  ┌─────────────────────┐  │  │  ┌───────────────────┐  │          │
│            │  │ MegatronFLEngine    │  │  │  │ vLLM + plugin-FL  │  │          │
│            │  │ ├─ TransformerEngine│  │  │  │ ├─ vllm_plugin-FL │  │          │
│            │  │ │  -FL (TE-FL)      │  │  │  │ ├─ FlagGems       │  │          │
│            │  │ ├─ FlagGems         │  │  │  │ └─ FlagCX         │  │          │
│            │  │ └─ FlagCX           │  │  │  └───────────────────┘  │          │
│            │  └─────────────────────┘  │  │                         │          │
│            │                           │  │  ┌───────────────────┐  │          │
│            │  ┌─────────────────────┐  │  │  │ SGLang + plugin   │  │          │
│            │  │ FSDPFLEngine        │  │  │  │ (future support)  │  │          │
│            │  │ ├─ FlagGems         │  │  │  └───────────────────┘  │          │
│            │  │ └─ FlagCX           │  │  │                         │          │
│            │  └─────────────────────┘  │  └─────────────────────────┘          │
│            └───────────────────────────┘                                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         External Dependencies                             │   │
│  │                                                                           │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │   │  FlagGems   │  │   FlagCX    │  │   TE-FL     │  │ vllm-plugin │    │   │
│  │   │  (Triton    │  │  (Comm      │  │ (Megatron   │  │    -FL      │    │   │
│  │   │  Operators) │  │   Library)  │  │  Training)  │  │  (Rollout)  │    │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  │                                                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. FLEnvManager

A lightweight environment variable manager that provides:
- Query methods for FL status (`is_fl_enabled()`, `is_flaggems_enabled()`, etc.)
- Separation of training and rollout phase environment variables
- FlagGems whitelist/blacklist retrieval per phase
- Summary string generation for logging

**Location**: `verl/utils/fl/config_manager.py`

### 2. Training Engines

#### MegatronFLEngineWithLMHead

Extends `MegatronEngineWithLMHead` with FL support through TransformerEngine-FL.

**Location**: `verl/workers/engine/megatron_fl/`

**Features**:
- Automatic TE-FL environment validation
- FlagGems operator integration
- FlagCX communication backend support

#### FSDPFLEngineWithLMHead / FSDPFLEngineWithValueHead

Extends FSDP engines with FL support through FlagGems operators.

**Location**: `verl/workers/engine/fsdp_fl/`

**Features**:
- FlagGems operator replacement for PyTorch native ops
- FlagCX communication support

### 3. Rollout Components

Rollout uses vllm-plugin-FL or sglang-plugin-FL for multi-chip inference support.

**Environment Control**: `VLLM_FL_*` environment variables

## Environment Variables

### Training Phase

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `TE_FL_PREFER` | TE-FL backend priority | `flagos` / `vendor` / `reference` | - |
| `TE_FL_STRICT` | Strict mode (no fallback) | `1` / `0` | `0` |
| `TE_FL_ALLOW_VENDORS` | Allowed vendors whitelist | `nvidia,amd` | - |
| `TE_FL_DENY_VENDORS` | Denied vendors blacklist | `vendor_a` | - |
| `TE_FL_PER_OP` | Per-operator configuration | `rmsnorm_fwd=vendor:cuda\|default` | - |
| `TEFL_LOG_LEVEL` | TE-FL log level | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |
| `TRAINING_FL_FLAGOS_WHITELIST` | FlagGems operator whitelist | `mm,bmm,softmax` | - |
| `TRAINING_FL_FLAGOS_BLACKLIST` | FlagGems operator blacklist | `layernorm` | - |

You can see a more detailed reference from [te-fl](https://github.com/flagos-ai/TransformerEngine-FL/pull/4), and the configuration may update.

### Rollout Phase

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `VLLM_FL_PREFER_ENABLED` | Enable FL preference | `true` / `false` | `false` |
| `VLLM_FL_PLATFORM` | Target platform | `cuda` / `npu` | `cuda` |
| `VLLM_FL_PREFER` | Backend priority | `flagos` / `vendor` | - |
| `VLLM_FL_OOT_ENABLED` | Enable out-of-tree plugins | `1` / `0` | `0` |
| `VLLM_FL_FLAGOS_WHITELIST` | FlagGems operator whitelist | `mm,bmm` | - |
| `VLLM_FL_FLAGOS_BLACKLIST` | FlagGems operator blacklist | `layernorm` | - |
| `VLLM_FL_CONFIG` | Dispatch config | `config file path` | - |

You can see a more detailed reference from [vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/dispatch/README.md#environment-variables), and the configuration may update.

### Common

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `USE_FLAGGEMS` | Enable FlagGems globally | `true` / `false` / `1` / `0` | `false` |
| `FLAGCX_PATH` | FlagCX installation path (setting this enables FlagCX) | `/path/to/FlagCX` | - |

## Engine Registration & Selection

FL engines are registered with `device="flagos"` and automatically selected based on environment variables:

```python
@EngineRegistry.register(model_type="language_model", backend="megatron", device="flagos")
class MegatronFLEngineWithLMHead(MegatronEngineWithLMHead):
    ...

@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="flagos")
class FSDPFLEngineWithLMHead(FSDPEngineWithLMHead):
    ...
```

**Automatic Engine Selection**: When `TE_FL_PREFER=flagos` is set, `EngineRegistry.get_engine_cls()` automatically selects FL engines instead of standard engines. No code changes or parameter passing required.

## Usage

All FL configurations are controlled through environment variables. No changes to verl command-line arguments or YAML configs are required.

### Setting Environment Variables

#### single node
```bash
#!/bin/bash

# ============ Training Phase Configuration ============
# TE-FL (TransformerEngine-FL) controls Megatron training kernels
export TE_FL_PREFER=flagos          # Backend: flagos / vendor / reference
export TE_FL_STRICT=0               # Strict mode (no fallback): 1 / 0
export TEFL_LOG_LEVEL=INFO          # Log level: DEBUG / INFO / WARNING / ERROR

# FlagGems operator control for training (optional)
# export TRAINING_FL_FLAGOS_WHITELIST=mm,bmm,softmax  # Only enable these ops
# export TRAINING_FL_FLAGOS_BLACKLIST=layernorm       # Disable these ops

# ============ Rollout Phase Configuration ============
# vLLM-FL controls inference kernels
export VLLM_FL_PREFER_ENABLED=true
export VLLM_FL_PLATFORM=cuda        # Platform: cuda / npu
export VLLM_FL_PREFER=flagos        # Backend: flagos / vendor
export VLLM_FL_OOT_ENABLED=1        # Enable out-of-tree plugins

# FlagGems operator control for rollout (optional)
# export VLLM_FL_FLAGOS_WHITELIST=mm,bmm
# export VLLM_FL_FLAGOS_BLACKLIST=layernorm

# ============ Common Configuration ============
export USE_FLAGGEMS=true            # Enable FlagGems globally
export FLAGCX_PATH=/path/to/FlagCX  # FlagCX installation path (enables FlagCX)

# ============ Run Training ============
python3 -m verl.trainer.main_ppo ...
```
#### multi nodes
Using ray runtime environment configuration.

```yaml
working_dir: ./
excludes: ["/.git/"]
env_vars:
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  TE_FL_PREFER: "flagos"
  TE_FL_STRICT: "0"
  TEFL_LOG_LEVEL: "INFO"
  ...
```
when you submit a job, specify the configuration yaml file
```bash
ray job submmit --runtime-env=<config file>

```

### Using FLEnvManager API (Optional)

`FLEnvManager` provides utility methods for querying FL configuration status within code:

```python
from verl.utils.fl import FLEnvManager

# Check FL status
FLEnvManager.is_fl_enabled()           # Any FL enabled?
FLEnvManager.is_training_fl_enabled()  # TE_FL_PREFER=flagos?
FLEnvManager.is_rollout_fl_enabled()   # VLLM_FL_PREFER=flagos?
FLEnvManager.is_flaggems_enabled()     # USE_FLAGGEMS=true?
FLEnvManager.is_flagcx_enabled()       # FLAGCX_PATH set?

# Get environment variables by phase
training_env = FLEnvManager.get_training_env()
rollout_env = FLEnvManager.get_rollout_env()

# Get FlagGems whitelist/blacklist
whitelist = FLEnvManager.get_flaggems_whitelist(phase="training")
blacklist = FLEnvManager.get_flaggems_blacklist(phase="rollout")

# Get summary string for logging
print(FLEnvManager.get_summary())  # e.g., "FL[training(TE_FL=flagos), FlagGems, FlagCX]"
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              PPO Training Loop                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Environment Variables Set (before process start)                    │
│     └── TE_FL_*, VLLM_FL_*, USE_FLAGGEMS, FLAGCX_PATH                   │
│                                                                          │
│  2. Engine Initialization                                                │
│     ├── EngineRegistry reads TE_FL_PREFER, selects FL engine            │
│     ├── MegatronFLEngine / FSDPFLEngine initializes                     │
│     └── may_enable_flag_gems(phase="training") configures FlagGems      │
│                                                                          │
│  3. Training Step                                                        │
│     ├── Forward pass: TE-FL kernels / FlagGems operators                │
│     ├── Backward pass: TE-FL kernels / FlagGems operators               │
│     └── Communication: FlagCX backend (if FLAGCX_PATH set)              │
│                                                                          │
│  4. Rollout Step                                                         │
│     ├── vLLM reads VLLM_FL_* env vars for plugin dispatch               │
│     ├── Inference: vllm-plugin-FL with FlagGems                         │
│     └── Communication: FlagCX (if enabled)                              │
│                                                                          │
│  5. Repeat 3-4 for each iteration                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## External Dependencies

| Component | Repository | Purpose |
|-----------|-----------|---------|
| FlagGems | https://github.com/flagos-ai/FlagGems | Triton-based operators for multi-chip |
| FlagCX | https://github.com/flagos-ai/FlagCX | Multi-chip communication library |
| TransformerEngine-FL | https://github.com/flagos-ai/TransformerEngine-FL | Multi-chip training kernels |
| vllm-plugin-FL | https://github.com/flagos-ai/vllm-plugin-FL | Multi-chip inference support |

### TransformerEngine-FL Configuration

See: https://github.com/flagos-ai/TransformerEngine-FL/pull/4

TE-FL provides a dispatch mechanism for selecting operator implementations:
- `flagos`: Use FlagOS optimized kernels
- `vendor`: Use vendor-specific kernels (NVIDIA, AMD, etc.)
- `reference`: Use reference implementations

### vllm-plugin-FL Configuration

See: https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/dispatch/README.md

Installation: https://github.com/flagos-ai/vllm-plugin-FL/tree/main

vllm-plugin-FL provides out-of-tree plugin support for vLLM with multi-chip backends.

## File Structure

```
verl/
├── utils/
│   └── fl/
│       ├── __init__.py           # Module entry point
│       └── env_manager.py        # FLEnvManager implementation
├── workers/
│   └── engine/
│       ├── base.py               # EngineRegistry with FL device support
│       ├── megatron_fl/
│       │   ├── __init__.py
│       │   └── transformer_impl.py  # MegatronFLEngineWithLMHead
│       └── fsdp_fl/
│           ├── __init__.py
│           └── transformer_impl.py  # FSDPFLEngineWithLMHead/ValueHead
└── docs/
    └── flagos/
        └── fl_multi_chip_support.md  # This document
```

## Example Scripts

### FSDP + FL Training

```bash
examples/grpo_trainer/run_qwen3-0.6b_fl.sh
```

### Megatron + FL Training

```bash
examples/grpo_trainer/run_qwen3-0.6b_megatron_fl.sh
```

## Troubleshooting

### 1. FL Engine Not Selected

**Symptom**: Regular engine is used instead of FL engine.

**Solution**: Ensure `TE_FL_PREFER=flagos` is set before starting training.

### 2. FlagGems Import Error

**Symptom**: `ImportError: No module named 'flag_gems'`

**Solution**: Install FlagGems, see https://github.com/flagos-ai/FlagGems

### 3. FlagCX Communication Error

**Symptom**: Communication failures with FlagCX backend.

**Solution**:
1. Verify `FLAGCX_PATH` points to valid FlagCX installation
2. Check FlagCX torch plugin is in `PYTHONPATH`
3. Verify FlagCX is properly installed and compiled for your hardware

### 4. TE-FL Kernel Fallback

**Symptom**: TE-FL falls back to reference implementation.

**Solution**:
1. Set `TE_FL_STRICT=1` to disable fallback and see errors
2. Check `TEFL_LOG_LEVEL=DEBUG` for detailed logs
3. Verify TE-FL installation matches your hardware
