#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1024
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=truncate



set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6

# Threading to match DataLoader workers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "SCRATCH = $SCRATCH"
echo "TMPDIR  = $TMPDIR"

mkdir -p "$SCRATCH/ds"
rsync -a /cluster/home/nkompalla/Bachelorarbeit/ba_code/data/*.h5 "$SCRATCH/ds/" # HIER LOCAL-NODE PATH ZU DATEN #
ls -lh "$SCRATCH/ds"

# ---- Paths (ADAPTED TO YOUR SETUP) ----
SRC_DIR="$SCRATCH/ds"                       # where you staged the .h5 files
JOB_DIR="$SCRATCH/jobs/$SLURM_JOB_ID"       # persistent outputs
LOCAL_DIR="$TMPDIR/ds"                      # node-local fast storage

TRAIN_BN="pattern_paths_training_44_44.h5"
TEST_BN="pattern_paths_test_44_44.h5"

SRC_TRAIN="$SRC_DIR/$TRAIN_BN"
SRC_TEST="$SRC_DIR/$TEST_BN"

LOCAL_TRAIN="$LOCAL_DIR/$TRAIN_BN"
LOCAL_TEST="$LOCAL_DIR/$TEST_BN"

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

# Make dirs
mkdir -p "$LOCAL_DIR" "$JOB_DIR" logs checkpoints results

# Sanity checks
test -f "$SRC_TRAIN" || { echo "Missing: $SRC_TRAIN"; exit 2; }
test -f "$SRC_TEST"  || { echo "Missing: $SRC_TEST";  exit 2; }
df -h "$TMPDIR" || true

# Stage to node-local SSD
echo "Copying datasets to node-local SSD…"
rsync -a --info=progress2 "$SRC_TRAIN" "$LOCAL_TRAIN"
rsync -a --info=progress2 "$SRC_TEST"  "$LOCAL_TEST"

# Always sync results back (even on error/timeout)
finish() {
  status=$?
  echo "Syncing outputs to $JOB_DIR (status=$status)…"
  rsync -a checkpoints/ "$JOB_DIR/checkpoints/" || true
  rsync -a results/     "$JOB_DIR/results/"     || true
  rsync -a logs/        "$JOB_DIR/logs/"        || true
  exit $status
}
trap finish EXIT

echo "[INFO] Environment Setup finished. Starting Python."
# ---- Run your training using the *local* fast paths ----
# Adjust CLI flags to your script (examples below).
# If your train.py takes a single --data, use the training file; if it also needs test path, add it.
# python -u src/train_gpu_opti.py \
  # --data-train "$LOCAL_TRAIN" \
  # --data-test  "$LOCAL_TEST" \
  # --num-workers $SLURM_CPUS_PER_TASK

python -u src/lightning.py

# trap handles syncing back
