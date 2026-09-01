#!/usr/bin/env bash
# Relax six independent parents with base UMA, generate 48 local candidates,
# and freeze a score-independent 45-record C0-local tranche.

#SBATCH --job-name=uma-c0-local
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/user140002/logs/uma-c0-local-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-c0-local-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/local_candidates_v1/${SLURM_JOB_ID}}"
SEED_CONFIG="$REPO_ROOT/configs/sampling/calibration_local_relax_seeds_v1.yaml"
PORTFOLIO_CONFIG="$REPO_ROOT/configs/sampling/calibration_local_portfolio_45_v1.yaml"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$SEED_CONFIG" "$PORTFOLIO_CONFIG" > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    '$PYTHON' -m uma_pyscf.cli.main sample \
      '$SEED_CONFIG' --output-dir '$RUN_ROOT/seeds'
    '$PYTHON' validation/uma_sampling/relax_local_seeds.py \
      --manifest '$RUN_ROOT/seeds/calibration_local_relax_seeds_v1_candidates.json' \
      --output-dir '$RUN_ROOT/relaxed' \
      --model-cache-dir '$MODEL_CACHE_DIR'
    '$PYTHON' -m uma_pyscf.cli.main sample \
      '$RUN_ROOT/relaxed/sampling_config.json' --output-dir '$RUN_ROOT/candidates'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio \
      '$PORTFOLIO_CONFIG' \
      --source-root '$RUN_ROOT/candidates' \
      --output-dir '$RUN_ROOT/portfolio'
  "

echo "C0-local candidate artifacts: $RUN_ROOT"
