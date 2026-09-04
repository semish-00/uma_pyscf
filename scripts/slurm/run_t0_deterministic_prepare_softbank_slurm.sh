#!/usr/bin/env bash
# Build the local, scan, and connected high-energy sources for fixed T0.

#SBATCH --job-name=t0-det-prep
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/user140002/logs/t0-det-prep-%j.out
#SBATCH --error=/lustre/user140002/logs/t0-det-prep-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh}"
PYTHON="${PYTHON:-/lustre/user140002/python/fairchem-2.22-py312-v2/bin/python}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/test/t0_deterministic_v1/${SLURM_JOB_ID}}"
LOCAL="$REPO_ROOT/configs/sampling/t0_dimer_local_pool_v1.yaml"
LOCAL_PORTFOLIO="$REPO_ROOT/configs/sampling/t0_dimer_local_portfolio_40_v1.yaml"
SCAN="$REPO_ROOT/configs/sampling/t0_dimer_scan_48_v1.yaml"
HIGH="$REPO_ROOT/configs/sampling/t0_dimer_high_energy_18_v1.yaml"

mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$LOCAL" "$LOCAL_PORTFOLIO" "$SCAN" "$HIGH" \
  > "$RUN_ROOT/provenance/inputs.sha256"
srun --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT/src'
    '$PYTHON' -m uma_pyscf.cli.main sample '$LOCAL' --output-dir '$RUN_ROOT/local_pool'
    '$PYTHON' -m uma_pyscf.cli.main assemble-portfolio '$LOCAL_PORTFOLIO' \
      --source-root '$RUN_ROOT/local_pool' --output-dir '$RUN_ROOT/local'
    '$PYTHON' -m uma_pyscf.cli.main sample '$SCAN' --output-dir '$RUN_ROOT/scan'
    '$PYTHON' -m uma_pyscf.cli.main sample '$HIGH' --output-dir '$RUN_ROOT/high_energy'
  "
sha256sum \
  "$RUN_ROOT/local/t0_dimer_local_40_v1_candidates.json" \
  "$RUN_ROOT/scan/t0_dimer_scan_48_v1_candidates.json" \
  "$RUN_ROOT/high_energy/t0_dimer_high_energy_18_v1_candidates.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"
echo "T0 deterministic source artifacts: $RUN_ROOT"
