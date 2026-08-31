#!/usr/bin/env bash
#SBATCH --job-name=uma-predict-pool
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/user140002/runs/slurm/%x-%j.out
#SBATCH --error=/lustre/user140002/runs/slurm/%x-%j.err

set -euo pipefail

: "${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST to a versioned candidate manifest}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
CONTAINER_SHA256_FILE="${CONTAINER_SHA256_FILE:-${CONTAINER_IMAGE}.sha256}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/evaluation/mf_pfp_candidate_pool_base_uma_s_1p2_v1.yaml}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/uma/mf_pfp_candidate_prediction/${SLURM_JOB_ID}}"

mkdir -p /lustre/user140002/runs/slurm "$RUN_ROOT"
sha256sum "$CONFIG" "$CANDIDATE_MANIFEST" > "$RUN_ROOT/input_sha256.txt"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    '$PYTHON' -m uma_pyscf.cli.main predict-uma \
      --config '$CONFIG' \
      --manifest '$CANDIDATE_MANIFEST' \
      --model-cache-dir '$MODEL_CACHE_DIR' \
      --output '$RUN_ROOT/predictions.json' \
      --repository '$REPO_ROOT' \
      --container-sha256-file '$CONTAINER_SHA256_FILE'
  "
