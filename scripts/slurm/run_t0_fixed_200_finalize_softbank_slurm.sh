#!/usr/bin/env bash
# Assemble all completed T0-only source manifests into fixed T0-200.

#SBATCH --job-name=t0-200-freeze
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/user140002/logs/t0-200-freeze-%j.out
#SBATCH --error=/lustre/user140002/logs/t0-200-freeze-%j.err

set -euo pipefail
: "${T0_DETERMINISTIC_ROOT:?Set T0_DETERMINISTIC_ROOT}"
: "${T0_MD_ROOT:?Set T0_MD_ROOT}"
: "${T0_INTERPOLATION_ROOT:?Set T0_INTERPOLATION_ROOT}"
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
CONFIG="$REPO_ROOT/configs/sampling/t0_fixed_200_v1.yaml"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/test/t0_fixed_200_v1/${SLURM_JOB_ID}}"
SOURCE_ROOT="$RUN_ROOT/sources"

mkdir -p "$RUN_ROOT/provenance" "$SOURCE_ROOT" /lustre/user140002/logs
cp "$T0_DETERMINISTIC_ROOT/local/t0_dimer_local_40_v1_candidates.json" "$SOURCE_ROOT/"
cp "$T0_DETERMINISTIC_ROOT/scan/t0_dimer_scan_48_v1_candidates.json" "$SOURCE_ROOT/"
cp "$T0_DETERMINISTIC_ROOT/high_energy/t0_dimer_high_energy_18_v1_candidates.json" \
  "$SOURCE_ROOT/"
cp "$T0_INTERPOLATION_ROOT/t0_hcl_interpolation_18_v1_candidates.json" "$SOURCE_ROOT/"
cp "$T0_MD_ROOT/portfolio/t0_dimer_md_76_v1_candidates.json" "$SOURCE_ROOT/"
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$CONFIG" "$SOURCE_ROOT"/*.json > "$RUN_ROOT/provenance/inputs.sha256"
srun --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio '$CONFIG' \
      --source-root '$SOURCE_ROOT' --output-dir '$RUN_ROOT/portfolio'
  "
sha256sum \
  "$RUN_ROOT/portfolio/t0_fixed_200_v1_candidates.json" \
  "$RUN_ROOT/portfolio/t0_fixed_200_v1_portfolio_report.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"
echo "Frozen T0-200 artifacts: $RUN_ROOT"
