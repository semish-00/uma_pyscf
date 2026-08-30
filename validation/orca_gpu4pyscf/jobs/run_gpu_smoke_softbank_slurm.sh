#!/bin/bash
# Run Workstream A1/A2 and C0 checks in the pinned SoftBank GPU environment.
#
# Submit from the repository root:
#   mkdir -p /lustre/user140002/logs /lustre/user140002/artifacts
#   sbatch validation/orca_gpu4pyscf/jobs/run_gpu_smoke_softbank_slurm.sh

#SBATCH --job-name=uma-gpu-smoke
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-smoke-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-smoke-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
ARTIFACT_BASE="${ARTIFACT_BASE:-/lustre/user140002/artifacts/gpu4pyscf-a1-a2}"
VALIDATION_ROOT="${REPO_ROOT}/validation/orca_gpu4pyscf"
GPU_SMOKE_CHECK="${GPU_SMOKE_CHECK:-${VALIDATION_ROOT}/gpu_smoke_check.py}"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${SLURM_JOB_ID}"

for path in "$REPO_ROOT" "$CONTAINER_IMAGE" "$PACKAGE_ROOT" "$GPU_SMOKE_CHECK"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$ARTIFACT_ROOT" /lustre/user140002/.cache/cupy

git -C "$REPO_ROOT" rev-parse HEAD > "${ARTIFACT_ROOT}/git_commit.txt"
git -C "$REPO_ROOT" status --short --branch > "${ARTIFACT_ROOT}/git_status.txt"

export ARTIFACT_ROOT GPU_SMOKE_CHECK PACKAGE_ROOT VALIDATION_ROOT
export PYTHONPATH="$PACKAGE_ROOT:$VALIDATION_ROOT"
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$VALIDATION_ROOT" \
  bash -lc '
    set -euo pipefail

    python3 collect_environment.py \
      --output "$ARTIFACT_ROOT/environment.yaml"
    python3 "$GPU_SMOKE_CHECK" \
      --grid-level 3 \
      --nlc-grid-level 1 \
      --output "$ARTIFACT_ROOT/gpu_smoke_check.json"
    python3 run_suite.py suites/si_ge_h_cl_ladder_v1.json \
      --device gpu \
      --dry-run > "$ARTIFACT_ROOT/full_ladder_dry_run.log"
  '

echo "A1/A2/C0 artifacts: $ARTIFACT_ROOT"
