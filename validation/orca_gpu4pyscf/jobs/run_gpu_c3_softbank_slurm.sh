#!/bin/bash
# Run the one-axis-at-a-time C3 GPU4PySCF settings matrix on SoftBank AI DC.

#SBATCH --job-name=uma-gpu-c3
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-c3-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-c3-%j.err

set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/lustre/user140002/runtime/uma_pyscf}"
C3_ROOT="${C3_ROOT:-/lustre/user140002/runtime/uma_pyscf-c3}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
ARTIFACT_BASE="${ARTIFACT_BASE:-/lustre/user140002/artifacts/gpu4pyscf-c3}"
SUITE="${C3_ROOT}/suites/gpu_c3_settings_matrix_v1.json"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${SLURM_JOB_ID}"

for path in \
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

sha256sum \
  "$RUNTIME_ROOT/common.py" \
  "$RUNTIME_ROOT/run_pyscf.py" \
  "$RUNTIME_ROOT/run_suite.py" \
  > "${ARTIFACT_ROOT}/runtime_files.sha256"
find "$C3_ROOT/configs" "$C3_ROOT/structures" "$C3_ROOT/suites" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ARTIFACT_ROOT}/c3_inputs.sha256"

export ARTIFACT_ROOT C3_ROOT PACKAGE_ROOT RUNTIME_ROOT
export PYTHONPATH="$PACKAGE_ROOT:$RUNTIME_ROOT"
export PYTHONUNBUFFERED=1
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$C3_ROOT" \
  bash -lc '
    set -euo pipefail

    python3 "$RUNTIME_ROOT/run_suite.py" \
      "$C3_ROOT/suites/gpu_c3_settings_matrix_v1.json" \
      --device gpu \
      --root "$C3_ROOT" \
      --case-timeout-minutes 45 \
      --summary-output "$ARTIFACT_ROOT/session.json"
  '

echo "C3 artifacts: $ARTIFACT_ROOT"
