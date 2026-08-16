#!/usr/bin/env bash
#SBATCH --job-name=bpa3-gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs models

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 GPU training job ===\n'
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

printf '\n=== Pixi environment ===\n'
# The BPA3 environment is managed with Pixi. We use `pixi run ...` for all
# Python commands, so the job does not depend on an already active shell.
# Make sure your Pixi/PyTorch installation actually supports GPUs.
which pixi
pixi --version
pixi run python --version

printf '\n=== GPU visibility ===\n'
nvidia-smi || true
pixi run python - <<'GPU_CHECK_PY' || true
try:
    import torch
    print('torch:', torch.__version__)
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('Torch GPU check skipped/failed:', repr(exc))
GPU_CHECK_PY

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py training.py try_agent.py

RUN_DIR="runs/gpu-train-${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

printf '\n=== Training ===\n'
# Adapt this command to your training.py.
CMD="${BPA3_TRAIN_CMD:-pixi run python training.py}"
echo "Running training command: ${CMD}"
eval "${CMD}"

printf '\n=== Post-training checks ===\n'
if [[ -f models/model.obj ]]; then
    ls -lh models/model.obj
    pixi run python try_agent.py
else
    echo "WARNING: models/model.obj was not found after training."
fi

printf '\nOptional GPU job finished.\n'
