#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --time=0:15:00
#SBATCH --array=1-20
#SBATCH --job-name="train" 
#SBATCH --mem-per-cpu=4096 
#SBATCH --output="logs/%x_%j.out" 
#SBATCH --error="logs/%x_%j.err" 
#SBATCH --open-mode=truncate

set -euo pipefail

# --- Modules / env ---
module load stack/2024-06 python_cuda/3.11.6
# If you use a venv/conda, activate it here
# source ~/venvs/torch311/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/nkompalla/Bachelorarbeit/ba_code

# Ensure logs dir exists
mkdir -p logs

# Start GPU sampler in background (CSV every 1s)
GPULOG="logs/gpu_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv -l 1 > "$GPULOG" 2>/dev/null &
SMI_PID=$!

# Always stop sampler on exit, even on error
cleanup() {
  kill $SMI_PID >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Run training (resource-bound by Slurm)
# Tip: use srun for correct binding; remove if you prefer direct python
srun --cpu-bind=cores python -u src/train_gpu_opti.py 2>&1 | tee "logs/train_${SLURM_JOB_ID}.live.log"
PY_STATUS=${PIPESTATUS[0]}

# Final snapshot
echo "=== NVIDIA-SMI (after) ==="
nvidia-smi || true

exit $PY_STATUS
