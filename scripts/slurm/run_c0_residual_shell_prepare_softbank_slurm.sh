#!/usr/bin/env bash
# Freeze the two small, group-preserving C0 residual-shell manifests.

#SBATCH --job-name=c0-residual-prep
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/user140002/logs/c0-residual-prep-%j.out
#SBATCH --error=/lustre/user140002/logs/c0-residual-prep-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_residual_shell_v1/${SLURM_JOB_ID}}"
SCAN_CONFIG="$REPO_ROOT/configs/sampling/c0_residual_scan_shell_v1.yaml"
DISTORT_CONFIG="$REPO_ROOT/configs/sampling/c0_residual_distortion_shell_v1.yaml"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$SCAN_CONFIG" "$DISTORT_CONFIG" \
  "$REPO_ROOT"/configs/sampling/structures/c0_residual_shell_seeds_v1/*.xyz \
  "$CONTAINER_IMAGE.sha256" > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    python3 -m uma_pyscf.cli.main sample '$SCAN_CONFIG' --output-dir '$RUN_ROOT/scan'
    python3 -m uma_pyscf.cli.main sample '$DISTORT_CONFIG' --output-dir '$RUN_ROOT/distortion'
  "

sha256sum \
  "$RUN_ROOT/scan/c0_residual_scan_shell_v1_candidates.json" \
  "$RUN_ROOT/scan/c0_residual_scan_shell_v1_geometry_qc.json" \
  "$RUN_ROOT/distortion/c0_residual_distortion_shell_v1_candidates.json" \
  "$RUN_ROOT/distortion/c0_residual_distortion_shell_v1_geometry_qc.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "C0 residual-shell candidate artifacts: $RUN_ROOT"
