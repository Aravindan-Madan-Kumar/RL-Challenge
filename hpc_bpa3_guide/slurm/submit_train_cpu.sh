#!/usr/bin/env bash
#SBATCH --job-name=bpa3-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs models

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 CPU training job ===\n'
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

printf '\n=== Pixi environment ===\n'
# The BPA3 environment is managed with Pixi. We use `pixi run ...` for all
# Python commands, so the job does not depend on an already active shell.
which pixi
pixi --version
pixi run python --version

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py training.py try_agent.py

RUN_DIR="runs/train-${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

printf '\n=== Training ===\n'
# Adapt this command to your training.py.
# Example if your script supports arguments:
#   export BPA3_TRAIN_CMD="pixi run python training.py --seed 0 --total-steps 1000000 --output-dir ${RUN_DIR}"
# If your script has all settings hard-coded, use:
#   export BPA3_TRAIN_CMD="pixi run python training.py"
CMD="${BPA3_TRAIN_CMD:-pixi run python training.py}"
echo "Running training command: ${CMD}"
eval "${CMD}"

printf '\n=== Post-training checks ===\n'
if [[ -f models/model.obj ]]; then
    ls -lh models/model.obj
    pixi run python try_agent.py
else
    echo "WARNING: models/model.obj was not found after training."
    echo "Make sure your training code saves the final submission model to models/model.obj."
fi

printf '\nTraining job finished.\n'
