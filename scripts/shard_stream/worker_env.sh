#!/usr/bin/env bash

# Source this file before running a compressed Net4 LR-screen segment.

export V211_LOCAL_ROOT=${V211_LOCAL_ROOT:-/tmp/anfazsha-v211-lr-screen}
export V211_REPO=${V211_REPO:-/student/anfazsha/nnue-pytorch}
export VIRTUAL_ENV_DISABLE_PROMPT=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$V211_LOCAL_ROOT/tmp"
export PIP_CACHE_DIR="$V211_LOCAL_ROOT/pip-cache"
export TORCH_HOME="$V211_LOCAL_ROOT/torch-cache"
export TORCHINDUCTOR_CACHE_DIR="$V211_LOCAL_ROOT/inductor-cache"
export XDG_CACHE_HOME="$V211_LOCAL_ROOT/xdg-cache"
export CUDA_CACHE_PATH="$V211_LOCAL_ROOT/cuda-cache"

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$CUDA_CACHE_PATH" \
  "$V211_LOCAL_ROOT/work"

# shellcheck disable=SC1091
source "$V211_LOCAL_ROOT/venv/bin/activate"
cd "$V211_REPO"
