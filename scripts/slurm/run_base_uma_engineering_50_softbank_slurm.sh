#!/usr/bin/env bash
#SBATCH --job-name=uma-base-50
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/user140002/runs/slurm/%x-%j.out
#SBATCH --error=/lustre/user140002/runs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
CONTAINER_SHA256_FILE="${CONTAINER_SHA256_FILE:-${CONTAINER_IMAGE}.sha256}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
DATASET_DIR="${DATASET_DIR:-/lustre/user140002/runs/label/engineering_50_v1/1797134/dataset/ds_sigehcl_001}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/uma/base_eval_engineering_50_v1/${SLURM_JOB_ID}}"

mkdir -p /lustre/user140002/runs/slurm "$RUN_ROOT"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    '$PYTHON' -c 'from importlib.metadata import version; import torch; assert torch.cuda.is_available(); print(version(\"fairchem-core\"), torch.__version__, torch.cuda.get_device_name())'
    '$PYTHON' -m uma_pyscf.cli.main evaluate-uma \
      --config '$REPO_ROOT/configs/evaluation/engineering_50_base_uma_s_1p2_v1.yaml' \
      --dataset-dir '$DATASET_DIR' \
      --model-cache-dir '$MODEL_CACHE_DIR' \
      --output '$RUN_ROOT/evaluation.json' \
      --repository '$REPO_ROOT' \
      --container-sha256-file '$CONTAINER_SHA256_FILE'
  "
