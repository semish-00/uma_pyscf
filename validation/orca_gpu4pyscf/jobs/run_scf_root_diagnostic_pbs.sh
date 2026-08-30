#!/bin/bash
# Run one CPU SCF-root diagnostic with an explicit initial density.

#PBS -N scf_root_cpu
#PBS -j oe
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:mem=32gb

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

RUNTIME_ROOT="${RUNTIME_ROOT:-/home/seki/uma_pyscf_runtime}"
CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
INIT_GUESS="${INIT_GUESS:-minao}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/seki/miniconda3/envs/qcthermo/bin/python}"

mkdir -p "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS="${PBS_NCPUS:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$RUNTIME_ROOT"

"$PYTHON_EXECUTABLE" "$RUNTIME_ROOT/diagnose_scf_root.py" "$CONFIG" \
  --device cpu --init-guess "$INIT_GUESS" --output "$OUTPUT"
