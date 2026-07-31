#!/bin/bash
# Control arm for the prompt study: the open models on the same 476 prompts.
#
#   sbatch prompt_study/gen_prompts.slurm --out ~/Code-Smells/prompt_study/generations_qwen.jsonl
#   GEN_MODEL=deepseek-ai/deepseek-coder-1.3b-instruct \
#     sbatch prompt_study/gen_prompts.slurm --out ~/Code-Smells/prompt_study/generations_deepseek.jsonl
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=prompt-gen
#SBATCH --output=/home/some5943/promptgen_%j.log

source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf

REPO=${REPO:-~/Code-Smells}
export GEN_MODEL=${GEN_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
export GEN_BATCH=${GEN_BATCH:-8}
export GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node: $(hostname)"
echo "model: $GEN_MODEL  (batch $GEN_BATCH, max_new $GEN_MAX_TOKENS)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/prompt_study"
python fetch_prompts.py          # the prompt set is fetched, not committed
python generate_local.py "$@"
