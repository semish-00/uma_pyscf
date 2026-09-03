#!/usr/bin/env bash
# Generate and freeze the score-independent C0 strong-distortion manifest.

#SBATCH --job-name=uma-c0-distort
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/user140002/logs/uma-c0-distort-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-c0-distort-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_strong_distortion_18_v1/${SLURM_JOB_ID}}"
SAMPLING_CONFIG="$REPO_ROOT/configs/sampling/c0_strong_distortion_v1.yaml"
PORTFOLIO_CONFIG="$REPO_ROOT/configs/sampling/c0_strong_distortion_portfolio_18_v1.yaml"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$SAMPLING_CONFIG" "$PORTFOLIO_CONFIG" "$CONTAINER_IMAGE.sha256" \
  > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    python3 -m uma_pyscf.cli.main sample \
      '$SAMPLING_CONFIG' --output-dir '$RUN_ROOT/candidates'
    python3 -m uma_pyscf.cli.main assemble-portfolio \
      '$PORTFOLIO_CONFIG' \
      --source-root '$RUN_ROOT/candidates' \
      --output-dir '$RUN_ROOT/portfolio'
  "

sha256sum \
  "$RUN_ROOT/candidates/c0_strong_distortion_pool_v1_candidates.json" \
  "$RUN_ROOT/candidates/c0_strong_distortion_pool_v1_geometry_qc.json" \
  "$RUN_ROOT/portfolio/c0_strong_distortion_18_v1_candidates.json" \
  "$RUN_ROOT/portfolio/c0_strong_distortion_18_v1_portfolio_report.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "C0 strong-distortion candidate artifacts: $RUN_ROOT"
