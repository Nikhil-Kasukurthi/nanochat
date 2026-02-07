#!/bin/bash
set -e

# RunPod variant of speedrun.sh — trains a d16 LLM (pretraining + finetuning)
# on a single A100 PCIe GPU. Takes approximately 2-3 hours to complete.
#
# RunPod-specific fixes:
# - Adds ~/.local/bin to PATH (where uv installs on RunPod)
# - Sources uv env file for proper shell integration
# - Installs python3-dev (needed for torchao FP8 JIT CUDA compilation)
# - Installs vim/tmux for convenience during long training runs
# A100 notes:
# - No FP8 support (Ampere architecture)
# - No FA3 support, uses SDPA fallback with --window-pattern=L

# 1) Example launch (simplest):
# bash runs/runpod_speedrun.sh
# 2) Example launch in a screen session (because the run takes ~2-3 hours):
# screen -L -Logfile runs/runpod_speedrun.log -S speedrun bash runs/runpod_speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/runpod_speedrun.log -S speedrun bash runs/runpod_speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# RunPod system dependencies
# python3-dev: needed for torchao's FP8 CUDA extension JIT compilation (Python.h)
# vim/tmux: useful for inspecting logs and running sessions during long training runs
apt-get update -qq && apt-get install -y -qq python3-dev vim tmux 2>/dev/null

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed), then source its env for proper shell integration
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi
# ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash runs/runpod_speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 100 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d16 model on single A100 (no FP8, use full attention for SDPA efficiency)
python -m scripts.base_train --depth=24 --target-param-data-ratio=10 --device-batch-size=16 --total-batch-size=131072 --window-pattern=L --run=$WANDB_RUN --save-every=1000 --eval-every=500 --model-tag=d16
# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m scripts.base_eval --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
python -m scripts.chat_sft -- --device-batch-size=16 --total-batch-size=32768 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
