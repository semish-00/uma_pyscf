#!/bin/bash
# Build the pinned GPU4PySCF Python package directory on SoftBank AI DC.
#
# Submit from the repository root after creating the log directory:
#   mkdir -p /lustre/user140002/logs
#   sbatch validation/orca_gpu4pyscf/jobs/setup_gpu4pyscf_softbank_slurm.sh

#SBATCH --job-name=uma-gpu-env
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-env-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-env-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
REQUIREMENTS="${REQUIREMENTS:-${REPO_ROOT}/validation/orca_gpu4pyscf/setup/requirements-gpu-cu122.in}"
LOCK_OUTPUT="${LOCK_OUTPUT:-${PACKAGE_ROOT}.lock.txt}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/lustre/user140002/.cache/pip}"

for path in "$REPO_ROOT" "$CONTAINER_IMAGE" "$REQUIREMENTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$PACKAGE_ROOT")" "$PIP_CACHE_DIR"

export PACKAGE_ROOT REQUIREMENTS LOCK_OUTPUT PIP_CACHE_DIR

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  bash -lc '
    set -euo pipefail

    if [[ -d "$PACKAGE_ROOT" ]]; then
      echo "package directory already exists; validating without mutation: $PACKAGE_ROOT"
      PYTHONPATH="$PACKAGE_ROOT" python3 -m pip check
      PYTHONPATH="$PACKAGE_ROOT" python3 -c \
        "import cupy, gpu4pyscf, pyscf; print(cupy.__version__, gpu4pyscf.__version__, pyscf.__version__)"
      exit 0
    fi

    staging="$(mktemp -d "${PACKAGE_ROOT}.staging.XXXXXX")"
    cleanup() {
      rm -rf -- "$staging"
    }
    trap cleanup EXIT

    python3 -m pip install \
      --disable-pip-version-check \
      --target "$staging" \
      --requirement "$REQUIREMENTS"

    PYTHONPATH="$staging" python3 -m pip check
    PYTHONPATH="$staging" python3 -c \
      "import cupy, gpu4pyscf, pyscf; print(cupy.__version__, gpu4pyscf.__version__, pyscf.__version__)"
    python3 -m pip list --path "$staging" --format=freeze \
      | LC_ALL=C sort > "$LOCK_OUTPUT"

    mv "$staging" "$PACKAGE_ROOT"
    trap - EXIT
    echo "installed immutable package directory: $PACKAGE_ROOT"
    echo "resolved lock: $LOCK_OUTPUT"
  '

sha256sum "$CONTAINER_IMAGE" > "${CONTAINER_IMAGE}.sha256"
echo "container checksum: ${CONTAINER_IMAGE}.sha256"
