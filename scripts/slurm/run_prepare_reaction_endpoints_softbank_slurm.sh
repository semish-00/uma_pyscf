#!/usr/bin/env bash
# Validate and relax the eight atom-mapped reaction endpoint pairs with base UMA.

#SBATCH --job-name=uma-rxn-endpoints
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/user140002/logs/uma-rxn-endpoints-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-rxn-endpoints-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
ENDPOINT_CONFIG="$REPO_ROOT/configs/reactions/si_ge_reaction_endpoints_v1.yaml"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/reaction_endpoints_v1/${SLURM_JOB_ID}}"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$ENDPOINT_CONFIG" > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    '$PYTHON' validation/reaction_paths/prepare_endpoints.py \
      --config '$ENDPOINT_CONFIG' \
      --output-dir '$RUN_ROOT' \
      --model-cache-dir '$MODEL_CACHE_DIR'
  "

echo "Reaction endpoint artifacts: $RUN_ROOT"
