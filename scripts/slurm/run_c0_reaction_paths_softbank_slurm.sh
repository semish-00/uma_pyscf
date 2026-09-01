#!/usr/bin/env bash
# Run four base-UMA CI-NEB paths and freeze the 36-record C0 path manifest.

#SBATCH --job-name=uma-c0-neb
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/user140002/logs/uma-c0-neb-%j.out
#SBATCH --error=/lustre/user140002/logs/uma-c0-neb-%j.err

set -euo pipefail

: "${ENDPOINT_RUN_ROOT:?Set ENDPOINT_RUN_ROOT to a completed reaction endpoint run}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1}"
PATH_CONFIG="$REPO_ROOT/configs/reactions/c0_independent_ci_neb_v1.yaml"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_independent_ci_neb_v1/${SLURM_JOB_ID}}"

test -f "$ENDPOINT_RUN_ROOT/summary.json"
mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum \
  "$PATH_CONFIG" "$ENDPOINT_RUN_ROOT/summary.json" \
  > "$RUN_ROOT/provenance/inputs.sha256"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    export FAIRCHEM_CACHE_DIR='$MODEL_CACHE_DIR'
    '$PYTHON' validation/reaction_paths/run_ci_neb.py \
      --config '$PATH_CONFIG' \
      --endpoint-run-root '$ENDPOINT_RUN_ROOT' \
      --output-dir '$RUN_ROOT' \
      --model-cache-dir '$MODEL_CACHE_DIR'
    '$PYTHON' -m uma_pyscf.cli.main import-trajectory \
      '$RUN_ROOT/trajectory_import_config.json' \
      --source-root '$RUN_ROOT' \
      --output-dir '$RUN_ROOT/candidates'
  "

sha256sum \
  "$RUN_ROOT/summary.json" \
  "$RUN_ROOT/trajectory_import_config.json" \
  "$RUN_ROOT/candidates/c0_independent_reaction_paths_36_v1_candidates.json" \
  "$RUN_ROOT/candidates/c0_independent_reaction_paths_36_v1_geometry_qc.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"

echo "C0 reaction-path artifacts: $RUN_ROOT"
