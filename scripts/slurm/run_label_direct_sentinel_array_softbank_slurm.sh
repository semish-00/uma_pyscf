#!/usr/bin/env bash
# Run direct-SCF diagnostic labels across a six-GPU array; no release QC.

#SBATCH --job-name=c0-direct-label
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --array=0-5
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/user140002/logs/c0-direct-label-%A_%a.out
#SBATCH --error=/lustre/user140002/logs/c0-direct-label-%A_%a.err

set -euo pipefail
: "${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST to the frozen sentinel manifest}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
DFT_CONFIG="${DFT_CONFIG:-$REPO_ROOT/configs/dft/omol_wb97mv_tzvpd_direct_sentinel_v1.yaml}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:?Missing Slurm array size}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_direct_sentinel_v1/${SLURM_ARRAY_JOB_ID}/direct_labels}"
SHARD_ROOT="$RUN_ROOT/shards/shard_${SHARD_INDEX}"
mkdir -p "$SHARD_ROOT/provenance" "$SHARD_ROOT/input" /lustre/user140002/logs
cp "$CANDIDATE_MANIFEST" "$SHARD_ROOT/input/candidates.json"
MANIFEST="$SHARD_ROOT/input/candidates.json"
git -C "$REPO_ROOT" status --short --branch > "$SHARD_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$SHARD_ROOT/provenance/git_commit.txt"
sha256sum "$MANIFEST" "$DFT_CONFIG" "$CONTAINER_IMAGE.sha256" "$PACKAGE_ROOT.lock.txt" \
  > "$SHARD_ROOT/provenance/inputs.sha256"
export PYTHONPATH="$PACKAGE_ROOT:$REPO_ROOT/src"
export PYTHONUNBUFFERED=1 CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset UMA_PYSCF_SCRATCH
srun --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" bash -lc "
    set -euo pipefail
    export PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUPY_ACCELERATORS=cutensor CUPY_CACHE_DIR='$CUPY_CACHE_DIR'
    python3 -m uma_pyscf.cli.main label --config '$DFT_CONFIG' --manifest '$MANIFEST' \
      --shard-count '$SHARD_COUNT' --shard-index '$SHARD_INDEX' \
      --output-dir '$SHARD_ROOT/label' --dry-run > '$SHARD_ROOT/label_plan.json'
    python3 -m uma_pyscf.cli.main label --config '$DFT_CONFIG' --manifest '$MANIFEST' \
      --shard-count '$SHARD_COUNT' --shard-index '$SHARD_INDEX' \
      --output-dir '$SHARD_ROOT/label'
  "
