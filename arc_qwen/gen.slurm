#!/bin/bash
# SLURM job for benchmark generation on ARC (htc GPU cluster).
#
# The model is a variable, so the same script produces every model's generations
# under identical conditions (same prompts, same decoding, same batch size) --
# which is what makes the models comparable.
#
#   sbatch arc_qwen/gen.slurm --limit 5 --out ~/Code-Smells/arc_qwen/generations_test.jsonl
#   sbatch arc_qwen/gen.slurm --out ~/Code-Smells/arc_qwen/generations.jsonl
#   GEN_MODEL=deepseek-ai/deepseek-coder-1.3b-instruct \
#     sbatch arc_qwen/gen.slurm --out ~/Code-Smells/arc_qwen/generations_deepseek.jsonl
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smell-gen
#SBATCH --output=/home/some5943/gen_%j.log

source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf

REPO=${REPO:-~/Code-Smells}
export GEN_MODEL=${GEN_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}

echo "node: $(hostname)"
echo "model: $GEN_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO/arc_qwen"
python generate.py "$@"
