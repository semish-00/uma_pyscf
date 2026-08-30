#!/usr/bin/env bash
#SBATCH --job-name=uma-dataset-50
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/user140002/logs/%x_%j.out
#SBATCH --error=/lustre/user140002/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
DATASET_OVERLAY="${DATASET_OVERLAY:-/lustre/user140002/python/ase-lmdb-py310-v1}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/label/engineering_50_v1/1797134}"
DATASET_ROOT="${DATASET_ROOT:-$RUN_ROOT/dataset/ds_sigehcl_001}"

mkdir -p /lustre/user140002/logs

git -C "$REPO_ROOT" diff --quiet
git -C "$REPO_ROOT" diff --cached --quiet
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc '
    set -euo pipefail

    if [[ ! -d "'"$DATASET_OVERLAY"'" ]]; then
      staging="'"$DATASET_OVERLAY"'.staging-'"$SLURM_JOB_ID"'"
      if [[ -e "$staging" ]]; then
        echo "Refusing existing staging overlay: $staging" >&2
        exit 1
      fi
      python3 -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --target "$staging" \
        ase==3.26.0 \
        ase-db-backends==0.11.0 \
        lmdb==1.7.3
      PYTHONPATH="$staging" python3 -c \
        "import ase, lmdb; from importlib.metadata import version; print(ase.__version__, lmdb.__version__, version(\"ase-db-backends\"))"
      python3 -m pip list --path "$staging" --format=freeze > "$staging/requirements.freeze.txt"
      mv "$staging" "'"$DATASET_OVERLAY"'"
    fi

    export PYTHONPATH="'"$DATASET_OVERLAY"':'"$REPO_ROOT"'/src"
    python3 -m uma_pyscf.cli.main dataset \
      --config configs/datasets/engineering_50_ase_lmdb_v1.yaml \
      --split "'"$RUN_ROOT"'/baseline/splits/engineering_50_baseline_split_v1.json" \
      --records "'"$RUN_ROOT"'/qc/records" \
      --output-dir "'"$DATASET_ROOT"'"

    python3 -m uma_pyscf.cli.main verify-dataset \
      --manifest "'"$DATASET_ROOT"'/dataset_manifest.json" \
      --records "'"$RUN_ROOT"'/qc/records" \
      --dataset-dir "'"$DATASET_ROOT"'"

    echo "git_commit='"$COMMIT"'"
    echo "dataset_root='"$DATASET_ROOT"'"
  '
