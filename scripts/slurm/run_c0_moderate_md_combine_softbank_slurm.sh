#!/usr/bin/env bash
# Freeze the parent-balanced 27-record moderate-MD portfolio from two pools.

#SBATCH --job-name=uma-c0-md-merge
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/user140002/logs/uma-c0-md-merge-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-c0-md-merge-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
PRIMARY_MANIFEST="${PRIMARY_MANIFEST:-/lustre/user140002/runs/calibration/c0_moderate_uma_md_v1/1825031/candidates/c0_moderate_md_pool_v1_candidates.json}"
RECOVERY_MANIFEST="${RECOVERY_MANIFEST:-/lustre/user140002/runs/calibration/c0_moderate_uma_md_ge2h4cl2_recovery_v1/1825383/candidates/c0_moderate_md_pool_v1_candidates.json}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_moderate_md_combined_27_v1/${SLURM_JOB_ID}}"
PORTFOLIO_CONFIG="$REPO_ROOT/configs/sampling/c0_moderate_md_combined_portfolio_27_v1.yaml"

mkdir -p "$RUN_ROOT/sources" "$RUN_ROOT/provenance" /lustre/user140002/logs
cp "$PRIMARY_MANIFEST" "$RUN_ROOT/sources/primary_candidates.json"
cp "$RECOVERY_MANIFEST" "$RUN_ROOT/sources/ge2h4cl2_recovery_candidates.json"
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum \
  "$PRIMARY_MANIFEST" "$RECOVERY_MANIFEST" "$PORTFOLIO_CONFIG" \
  "$CONTAINER_IMAGE.sha256" > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    python3 -m uma_pyscf.cli.main assemble-portfolio \
      '$PORTFOLIO_CONFIG' \
      --source-root '$RUN_ROOT/sources' \
      --output-dir '$RUN_ROOT/portfolio'
  "

sha256sum \
  "$RUN_ROOT/portfolio/c0_moderate_md_27_v1_candidates.json" \
  "$RUN_ROOT/portfolio/c0_moderate_md_27_v1_portfolio_report.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "C0 combined moderate-MD portfolio: $RUN_ROOT"
