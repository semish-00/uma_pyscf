#!/usr/bin/env bash
# Generate the acquisition-independent P0 pilot and freeze equal-budget
# random and model-independent D-optimal 24-record arms.

#SBATCH --job-name=p0-blind-prep
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/user140002/logs/p0-blind-prep-%j.out
#SBATCH --error=/lustre/user140002/logs/p0-blind-prep-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
POOL_CONFIG="$REPO_ROOT/configs/sampling/p0_blind_pilot_pool_v1.yaml"
RANDOM_CONFIG="$REPO_ROOT/configs/sampling/p0_blind_random_24_v1.yaml"
DOPT_CONFIG="$REPO_ROOT/configs/sampling/p0_blind_doptimal_24_v1.yaml"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/p0_blind_pilot_v1/${SLURM_JOB_ID}}"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$POOL_CONFIG" "$RANDOM_CONFIG" "$DOPT_CONFIG" \
  > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    '$PYTHON' -m uma_pyscf.cli.main sample \
      '$POOL_CONFIG' --output-dir '$RUN_ROOT/candidates'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio \
      '$RANDOM_CONFIG' \
      --source-root '$RUN_ROOT/candidates' \
      --output-dir '$RUN_ROOT/random'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio \
      '$DOPT_CONFIG' \
      --source-root '$RUN_ROOT/candidates' \
      --output-dir '$RUN_ROOT/doptimal'
  "

sha256sum \
  "$RUN_ROOT/candidates/p0_blind_pilot_pool_v1_candidates.json" \
  "$RUN_ROOT/candidates/p0_blind_pilot_pool_v1_geometry_qc.json" \
  "$RUN_ROOT/random/p0_blind_random_24_v1_candidates.json" \
  "$RUN_ROOT/random/p0_blind_random_24_v1_portfolio_report.json" \
  "$RUN_ROOT/doptimal/p0_blind_doptimal_24_v1_candidates.json" \
  "$RUN_ROOT/doptimal/p0_blind_doptimal_24_v1_portfolio_report.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "P0 blind pilot candidate artifacts: $RUN_ROOT"
