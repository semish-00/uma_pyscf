#!/usr/bin/env bash
# Relax six independent parents, run short base-UMA NVT trajectories, and
# freeze a score-independent 27-record C0 moderate-temperature MD tranche.

#SBATCH --job-name=uma-c0-md
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/user140002/logs/uma-c0-md-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-c0-md-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
SEED_CONFIG="$REPO_ROOT/configs/sampling/c0_moderate_md_parent_seeds_v1.yaml"
MD_CONFIG="$REPO_ROOT/configs/sampling/c0_moderate_uma_md_v1.yaml"
PORTFOLIO_CONFIG="$REPO_ROOT/configs/sampling/c0_moderate_md_portfolio_27_v1.yaml"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_moderate_uma_md_v1/${SLURM_JOB_ID}}"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$SEED_CONFIG" "$MD_CONFIG" "$PORTFOLIO_CONFIG" \
  > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    export OMP_NUM_THREADS='${SLURM_CPUS_PER_TASK:-1}'
    '$PYTHON' -m uma_pyscf.cli.main sample \
      '$SEED_CONFIG' --output-dir '$RUN_ROOT/seeds'
    '$PYTHON' validation/uma_sampling/run_langevin_md.py \
      --config '$MD_CONFIG' \
      --manifest '$RUN_ROOT/seeds/c0_moderate_md_parent_seeds_v1_candidates.json' \
      --output-dir '$RUN_ROOT/md' \
      --model-cache-dir '$MODEL_CACHE_DIR' \
      --keep-going
    '$PYTHON' -m uma_pyscf.cli.main import-trajectory \
      '$RUN_ROOT/md/trajectory_import_config.json' \
      --source-root '$RUN_ROOT/md' \
      --output-dir '$RUN_ROOT/candidates'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio \
      '$PORTFOLIO_CONFIG' \
      --source-root '$RUN_ROOT/candidates' \
      --output-dir '$RUN_ROOT/portfolio'
  "

sha256sum \
  "$RUN_ROOT/md/summary.json" \
  "$RUN_ROOT/candidates/c0_moderate_md_pool_v1_candidates.json" \
  "$RUN_ROOT/candidates/c0_moderate_md_pool_v1_geometry_qc.json" \
  "$RUN_ROOT/portfolio/c0_moderate_md_27_v1_candidates.json" \
  "$RUN_ROOT/portfolio/c0_moderate_md_27_v1_report.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "C0 moderate-temperature MD artifacts: $RUN_ROOT"
