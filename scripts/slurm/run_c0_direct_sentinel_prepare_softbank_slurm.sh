#!/usr/bin/env bash
# Freeze the six-record C0 direct-SCF sentinel manifest.

#SBATCH --job-name=c0-direct-prep
#SBATCH --partition=140-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/user140002/logs/c0-direct-prep-%j.out
#SBATCH --error=/lustre/user140002/logs/c0-direct-prep-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/lustre/user140002/uma_pyscf_calibration}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh}"
RUN_ROOT="${RUN_ROOT:-/lustre/user140002/runs/calibration/c0_direct_sentinel_v1/${SLURM_JOB_ID}}"
CONFIG="$REPO_ROOT/configs/sampling/c0_direct_sentinel_v1.yaml"
mkdir -p "$RUN_ROOT/provenance" /lustre/user140002/logs
git -C "$REPO_ROOT" status --short --branch > "$RUN_ROOT/provenance/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
sha256sum "$CONFIG" \
  "$REPO_ROOT/configs/dft/omol_wb97mv_tzvpd_direct_sentinel_v1.yaml" \
  "$REPO_ROOT/configs/sampling/structures/c0_source_diverse_shell_seeds_v1/local_ge2h6.xyz" \
  "$REPO_ROOT/configs/sampling/structures/c0_residual_shell_seeds_v1/scan_sih2cl2.xyz" \
  "$REPO_ROOT/configs/sampling/structures/c0_source_diverse_shell_seeds_v1/path_geh3cl_to_gehcl_h2.xyz" \
  "$REPO_ROOT/configs/sampling/structures/c0_source_diverse_shell_seeds_v1/md_ge2h4cl2.xyz" \
  "$REPO_ROOT/configs/sampling/structures/c0_residual_shell_seeds_v1/distort_si2h4cl2.xyz" \
  "$REPO_ROOT/configs/sampling/structures/c0_source_diverse_shell_seeds_v1/graph_ge2h4cl2_geminal.xyz" \
  "$CONTAINER_IMAGE.sha256" > "$RUN_ROOT/provenance/inputs.sha256"
srun --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" bash -lc "
    set -euo pipefail; export PYTHONPATH='$REPO_ROOT/src'
    python3 -m uma_pyscf.cli.main sample '$CONFIG' --output-dir '$RUN_ROOT/candidates'
  "
sha256sum "$RUN_ROOT/candidates/c0_direct_sentinel_v1_candidates.json" \
  "$RUN_ROOT/candidates/c0_direct_sentinel_v1_geometry_qc.json" \
  > "$RUN_ROOT/provenance/outputs.sha256"
