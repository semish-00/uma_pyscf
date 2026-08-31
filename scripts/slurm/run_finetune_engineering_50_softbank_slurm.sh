#!/usr/bin/env bash
#SBATCH --job-name=uma-overfit-50
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
PYTHON_ROOT="${PYTHON_ROOT:-/lustre/user140002/python/fairchem-2.22-py312-v2}"
DATASET_ROOT="${DATASET_ROOT:-/lustre/user140002/runs/label/engineering_50_v1/1797134/dataset/ds_sigehcl_001}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/uma/overfit_engineering_50_v1/${SLURM_JOB_ID}}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/finetune/engineering_50_overfit_v1.yaml}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"

mkdir -p /lustre/user140002/runs/slurm "$RUN_ROOT"
sha256sum "$CONFIG" "$REPO_ROOT/configs/finetune/data/engineering_50_omol_ef_v1.yaml" \
  "$DATASET_ROOT/dataset_manifest.json" > "$RUN_ROOT/input_sha256.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export PYTHONHASHSEED=41
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    export UMA_TRAIN_DATA='$DATASET_ROOT/train'
    export UMA_VAL_DATA='$DATASET_ROOT/holdout'
    export UMA_TRAIN_RUN_DIR='$RUN_ROOT'
    '$PYTHON_ROOT/bin/python' -c 'from importlib.metadata import version; import torch; assert torch.cuda.is_available(); print(version(\"fairchem-core\"), torch.__version__, torch.cuda.get_device_name())'
    '$PYTHON_ROOT/bin/fairchem' -c '$CONFIG' steps='$TRAIN_STEPS'
  "
