#!/bin/bash
# Run one GPU SCF-root diagnostic with the same CPU-generated initial density.

#SBATCH --job-name=uma-gpu-root
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/user140002/logs/uma-gpu-root-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-gpu-root-%j.err

set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/lustre/user140002/runtime/uma_pyscf}"
PROBE_ROOT="${PROBE_ROOT:-/lustre/user140002/runtime/uma_pyscf-spin}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
CONFIG="${CONFIG:-${PROBE_ROOT}/configs/geh3_neutral_quartet_density_fit_probe.json}"
OUTPUT="${OUTPUT:-${PROBE_ROOT}/diagnostics/geh3_neutral_quartet_gpu_minao.json}"

mkdir -p "$(dirname "$OUTPUT")"
export CONFIG OUTPUT PACKAGE_ROOT PROBE_ROOT RUNTIME_ROOT
export PYTHONPATH="$PACKAGE_ROOT:$RUNTIME_ROOT"
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$PROBE_ROOT" \
  bash -lc '
    set -euo pipefail
    python3 "$RUNTIME_ROOT/diagnose_scf_root.py" "$CONFIG" \
      --device gpu --init-guess minao --output "$OUTPUT"
  '
