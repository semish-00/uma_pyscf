#!/usr/bin/env bash
# Label one immutable candidate manifest with GPU4PySCF and run production QC.

#SBATCH --job-name=uma-label-manifest
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/user140002/logs/uma-label-manifest-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-label-manifest-%j.err

set -euo pipefail

: "${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST to an immutable candidate manifest}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
DFT_CONFIG="${DFT_CONFIG:-$REPO_ROOT/configs/dft/omol_wb97mv_tzvpd_v1.yaml}"
QC_CONFIG="${QC_CONFIG:-$REPO_ROOT/configs/datasets/omol_wb97mv_tzvpd_conditional_qc_v1.yaml}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/label/calibration_local_45_v1/${SLURM_JOB_ID}}"

mkdir -p "$RUN_ROOT/provenance" "$RUN_ROOT/input" /lustre/user140002/logs
cp "$CANDIDATE_MANIFEST" "$RUN_ROOT/input/candidates.json"
MANIFEST="$RUN_ROOT/input/candidates.json"
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum \
  "$MANIFEST" "$DFT_CONFIG" "$QC_CONFIG" \
  "$CONTAINER_IMAGE.sha256" "$PACKAGE_ROOT.lock.txt" \
  > "$RUN_ROOT/provenance/inputs.sha256"

export PYTHONPATH="$PACKAGE_ROOT:$REPO_ROOT/src"
export PYTHONUNBUFFERED=1
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset UMA_PYSCF_SCRATCH

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$PYTHONPATH'
    export PYTHONUNBUFFERED=1
    export CUPY_ACCELERATORS=cutensor
    export CUPY_CACHE_DIR='$CUPY_CACHE_DIR'
    python3 -m uma_pyscf.cli.main label \
      --config '$DFT_CONFIG' --manifest '$MANIFEST' \
      --output-dir '$RUN_ROOT/label' --dry-run > '$RUN_ROOT/label_plan.json'
    python3 -m uma_pyscf.cli.main label \
      --config '$DFT_CONFIG' --manifest '$MANIFEST' \
      --output-dir '$RUN_ROOT/label'
    python3 -m uma_pyscf.cli.main qc \
      --config '$QC_CONFIG' --records '$RUN_ROOT/label/records' \
      --output-dir '$RUN_ROOT/qc'
  "

echo "GPU4PySCF label artifacts: $RUN_ROOT"
