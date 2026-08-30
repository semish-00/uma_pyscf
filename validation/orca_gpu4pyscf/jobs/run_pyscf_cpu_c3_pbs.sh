#!/bin/bash
# Run one C3 PySCF CPU candidate in an isolated ujilab execution root.

#PBS -N uma_c3_cpu
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:mem=32gb

set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

RUNTIME_ROOT="${RUNTIME_ROOT:-/home/seki/uma_pyscf_runtime}"
CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/seki/miniconda3/envs/qcthermo/bin/python}"
THREADS="${THREADS:-8}"

case "$THREADS" in
  ''|*[!0-9]*|0)
    echo "THREADS must be a positive integer, got: $THREADS" >&2
    exit 2
    ;;
esac
for path in "$CONFIG" "$RUNTIME_ROOT/common.py" "$RUNTIME_ROOT/run_pyscf.py"; do
  if [[ ! -f "$path" ]]; then
    echo "required file does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "PYTHON_EXECUTABLE is not executable: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONPATH="$RUNTIME_ROOT"
export PYTHONUTF8=1
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export LANG="${LANG:-en_US.UTF-8}"

echo "CONFIG=$CONFIG"
echo "OUTPUT=$OUTPUT"
echo "PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE"
echo "THREADS=$THREADS"
"$PYTHON_EXECUTABLE" --version

"$PYTHON_EXECUTABLE" \
  "$RUNTIME_ROOT/run_pyscf.py" \
  "$CONFIG" \
  --device cpu \
  --output "$OUTPUT"
