#!/usr/bin/env bash
# Generate and label the 24-record Si2H3/Si2H5/Ge2H3/Ge2H5 C0-S matrix.

#SBATCH --job-name=uma-dimer-state-audit
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/user140002/logs/uma-dimer-state-audit-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-dimer-state-audit-%j.err

set -euo pipefail

: "${ENDPOINT_RUN_ROOT:?Set ENDPOINT_RUN_ROOT to a completed reaction endpoint run}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1}"
DFT_CONFIG="$REPO_ROOT/configs/dft/omol_wb97mv_tzvpd_state_audit_v1.yaml"
QC_CONFIG="$REPO_ROOT/configs/datasets/omol_wb97mv_tzvpd_state_audit_qc_v1.yaml"
CANDIDATE_CONFIG="$ENDPOINT_RUN_ROOT/state_audit/sampling_config.json"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/si_ge_dimer_state_audit_v1/${SLURM_JOB_ID}}"

test -f "$ENDPOINT_RUN_ROOT/summary.json"
test -f "$CANDIDATE_CONFIG"
mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum \
  "$ENDPOINT_RUN_ROOT/summary.json" "$CANDIDATE_CONFIG" "$DFT_CONFIG" "$QC_CONFIG" \
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
    python3 -m uma_pyscf.cli.main sample \
      '$CANDIDATE_CONFIG' --output-dir '$RUN_ROOT/candidates'
    MANIFEST='$RUN_ROOT/candidates/si_ge_dimer_state_audit_v1_candidates.json'
    python3 -m uma_pyscf.cli.main label \
      --config '$DFT_CONFIG' --manifest \"\$MANIFEST\" \
      --output-dir '$RUN_ROOT/label' --dry-run > '$RUN_ROOT/label_plan.json'
    python3 -m uma_pyscf.cli.main label \
      --config '$DFT_CONFIG' --manifest \"\$MANIFEST\" \
      --output-dir '$RUN_ROOT/label'
    python3 -m uma_pyscf.cli.main qc \
      --config '$QC_CONFIG' --records '$RUN_ROOT/label/records' \
      --output-dir '$RUN_ROOT/qc'
    python3 validation/reaction_paths/summarize_state_audit.py \
      --records-dir '$RUN_ROOT/label/records' \
      --output '$RUN_ROOT/state_audit_summary.json'
  "

echo "Si/Ge dimer state-audit artifacts: $RUN_ROOT"
