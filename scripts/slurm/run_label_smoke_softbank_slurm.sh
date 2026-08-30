#!/bin/bash
# End-to-end P2.3 engineering smoke: sample -> GPU label -> production QC.

#SBATCH --job-name=uma-p23-smoke
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/user140002/logs/uma-p23-smoke-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-p23-smoke-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
SAMPLE_CONFIG="${SAMPLE_CONFIG:-$REPO_ROOT/configs/sampling/gpu_label_smoke_v1.yaml}"
DFT_CONFIG="${DFT_CONFIG:-$REPO_ROOT/configs/dft/omol_wb97mv_tzvpd_v1.yaml}"
QC_CONFIG="${QC_CONFIG:-$REPO_ROOT/configs/datasets/omol_wb97mv_tzvpd_conditional_qc_v1.yaml}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/label/gpu_label_smoke_v1/$SLURM_JOB_ID}"

for path in \
  "$REPO_ROOT/src/uma_pyscf" \
  "$CONTAINER_IMAGE" \
  "$PACKAGE_ROOT" \
  "$SAMPLE_CONFIG" \
  "$DFT_CONFIG" \
  "$QC_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum \
  "$SAMPLE_CONFIG" \
  "$DFT_CONFIG" \
  "$QC_CONFIG" \
  "$CONTAINER_IMAGE.sha256" \
  "$PACKAGE_ROOT.lock.txt" \
  > "$RUN_ROOT/provenance/inputs.sha256"

export REPO_ROOT PACKAGE_ROOT SAMPLE_CONFIG DFT_CONFIG QC_CONFIG RUN_ROOT
export PYTHONPATH="$PACKAGE_ROOT:$REPO_ROOT/src"
export PYTHONUNBUFFERED=1
export CUPY_ACCELERATORS=cutensor
export CUPY_CACHE_DIR=/lustre/user140002/.cache/cupy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Each candidate uses an automatically cleaned TemporaryDirectory under the
# container-local /tmp on the allocated GPU node.
unset UMA_PYSCF_SCRATCH

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc '
    set -euo pipefail
    python3 -m uma_pyscf.cli.main sample \
      "$SAMPLE_CONFIG" \
      --output-dir "$RUN_ROOT/input"
    MANIFEST="$RUN_ROOT/input/gpu_label_smoke_v1_candidates.json"
    python3 -m uma_pyscf.cli.main label \
      --config "$DFT_CONFIG" \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_ROOT/label" \
      --dry-run \
      > "$RUN_ROOT/label_plan.json"
    python3 -m uma_pyscf.cli.main label \
      --config "$DFT_CONFIG" \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_ROOT/label"
    python3 -m uma_pyscf.cli.main qc \
      --config "$QC_CONFIG" \
      --records "$RUN_ROOT/label/records" \
      --output-dir "$RUN_ROOT/qc"
  '

echo "P2.3 smoke artifacts: $RUN_ROOT"
