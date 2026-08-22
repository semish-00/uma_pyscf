#!/bin/bash
# CPU PySCF energy/gradient validation on ujilab's PBS/Torque compute nodes.
# Review resource directives on the target host before first submission.
#
# Example:
#   qsub -v CONFIG=validation/orca_gpu4pyscf/configs/h2_wb97mv_def2tzvpd.json \
#     validation/orca_gpu4pyscf/jobs/run_pyscf_cpu_pbs.sh
#PBS -N uma_pyscf_cpu
#PBS -j oe
#PBS -o validation/orca_gpu4pyscf/runs/
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=16gb

set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

CONFIG="${CONFIG:-validation/orca_gpu4pyscf/configs/h2_wb97mv_def2tzvpd.json}"
OUTPUT="${OUTPUT:-validation/orca_gpu4pyscf/runs/h2_wb97mv_def2tzvpd/pyscf-cpu/result.json}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/seki/miniconda3/envs/qcthermo/bin/python}"
THREADS="${THREADS:-4}"

case "$THREADS" in
  ''|*[!0-9]*|0)
    echo "THREADS must be a positive integer, got: $THREADS" >&2
    exit 2
    ;;
esac
if [ ! -f "$CONFIG" ]; then
  echo "CONFIG does not exist: $CONFIG" >&2
  exit 2
fi
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
  echo "PYTHON_EXECUTABLE is not executable: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONUTF8=1
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export LANG="${LANG:-en_US.UTF-8}"

echo "CONFIG=$CONFIG"
echo "OUTPUT=$OUTPUT"
echo "PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE"
echo "THREADS=$THREADS"
"$PYTHON_EXECUTABLE" --version

"$PYTHON_EXECUTABLE" \
  validation/orca_gpu4pyscf/run_pyscf.py \
  "$CONFIG" \
  --device cpu \
  --output "$OUTPUT"
