#!/bin/bash
# Run the 12-state density-fitting charge/spin parity probe on SoftBank AI DC.

#SBATCH --job-name=uma-gpu-spin
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-spin-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-spin-%j.err

set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/lustre/user140002/runtime/uma_pyscf}"
PROBE_ROOT="${PROBE_ROOT:-/lustre/user140002/runtime/uma_pyscf-spin}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
ARTIFACT_BASE="${ARTIFACT_BASE:-/lustre/user140002/artifacts/gpu4pyscf-spin}"
SUITE_NAME="${SUITE_NAME:-charge_spin_density_fit_probe_v1.json}"
SUITE="${PROBE_ROOT}/suites/${SUITE_NAME}"
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
find "$PROBE_ROOT/configs" "$PROBE_ROOT/structures" "$PROBE_ROOT/suites" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ARTIFACT_ROOT}/probe_inputs.sha256"

export ARTIFACT_ROOT PACKAGE_ROOT PROBE_ROOT RUNTIME_ROOT SUITE_NAME
export PYTHONPATH="$PACKAGE_ROOT:$RUNTIME_ROOT"
export PYTHONUNBUFFERED=1
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$PROBE_ROOT" \
  bash -lc '
    set -euo pipefail
    python3 "$RUNTIME_ROOT/run_suite.py" \
      "$PROBE_ROOT/suites/$SUITE_NAME" \
      --device gpu \
      --root "$PROBE_ROOT" \
      --case-timeout-minutes 60 \
      --summary-output "$ARTIFACT_ROOT/session.json"
  '

echo "charge/spin artifacts: $ARTIFACT_ROOT"
