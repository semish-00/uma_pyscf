#!/bin/bash
# Run the five-case C1 GPU4PySCF smoke suite sequentially on SoftBank AI DC.

#SBATCH --job-name=uma-gpu-c1
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-c1-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-c1-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/lustre/user140002/runtime/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
ARTIFACT_BASE="${ARTIFACT_BASE:-/lustre/user140002/artifacts/gpu4pyscf-c1}"
VALIDATION_ROOT="${REPO_ROOT}/validation/orca_gpu4pyscf"
SUITE="${VALIDATION_ROOT}/suites/gpu_smoke_v1.json"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${SLURM_JOB_ID}"

for path in \
  "$REPO_ROOT" \
  "$RUNTIME_ROOT/common.py" \
  "$RUNTIME_ROOT/run_pyscf.py" \
  "$RUNTIME_ROOT/run_suite.py" \
  "$CONTAINER_IMAGE" \
  "$PACKAGE_ROOT" \
  "$SUITE"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$ARTIFACT_ROOT" /lustre/user140002/.cache/cupy

git -C "$REPO_ROOT" rev-parse HEAD > "${ARTIFACT_ROOT}/git_commit.txt"
git -C "$REPO_ROOT" status --short --branch > "${ARTIFACT_ROOT}/git_status.txt"
sha256sum \
  "$RUNTIME_ROOT/common.py" \
  "$RUNTIME_ROOT/run_pyscf.py" \
  "$RUNTIME_ROOT/run_suite.py" \
  > "${ARTIFACT_ROOT}/runtime_files.sha256"

export ARTIFACT_ROOT PACKAGE_ROOT RUNTIME_ROOT VALIDATION_ROOT
export PYTHONPATH="$PACKAGE_ROOT:$RUNTIME_ROOT:$VALIDATION_ROOT"
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$VALIDATION_ROOT" \
  bash -lc '
    set -euo pipefail

    python3 "$RUNTIME_ROOT/run_suite.py" \
      "$VALIDATION_ROOT/suites/gpu_smoke_v1.json" \
      --device gpu \
      --root "$VALIDATION_ROOT" \
      --case-timeout-minutes 360 \
      --summary-output "$ARTIFACT_ROOT/session.json"
  '

echo "C1 artifacts: $ARTIFACT_ROOT"
