#!/usr/bin/env bash
# ORCA 6.0.0 energy/gradient validation on ujilab OpenPBS.
#
# Submit from the repository root:
#   /usr/openpbs/bin/qsub \
#     validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh
#
# Optional path overrides:
#   /usr/openpbs/bin/qsub \
#     -v ORCA_ROOT=/home/seki/uma_pyscf/software/orca/6.0.0,OPENMPI_ROOT=/home/seki/uma_pyscf/software/openmpi/4.1.6 \
#     validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh
#PBS -N uma_orca_cpu
#PBS -q workq
#PBS -j oe
#PBS -o validation/orca_gpu4pyscf/runs/
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=16gb

set -euo pipefail

WORK_ROOT="${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
cd "$WORK_ROOT"

CONFIG="${CONFIG:-validation/orca_gpu4pyscf/configs/h2_wb97mv_def2tzvpd.json}"
RUN_DIR="${RUN_DIR:-validation/orca_gpu4pyscf/runs/h2_wb97mv_def2tzvpd/orca}"
ORCA_ROOT="${ORCA_ROOT:-/home/seki/uma_pyscf/software/orca/6.0.0}"
OPENMPI_ROOT="${OPENMPI_ROOT:-/home/seki/uma_pyscf/software/openmpi/4.1.6}"
ORCA_EXECUTABLE="${ORCA_EXECUTABLE:-${ORCA_ROOT}/orca}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/seki/miniconda3/envs/qcthermo/bin/python}"
KEEP_SCRATCH="${KEEP_SCRATCH:-0}"

if [ ! -f "$CONFIG" ]; then
  echo "CONFIG does not exist: $CONFIG" >&2
  exit 2
fi
if [ ! -x "$ORCA_EXECUTABLE" ]; then
  echo "Quantum-chemistry ORCA is not executable: $ORCA_EXECUTABLE" >&2
  exit 2
fi
ORCA_EXECUTABLE="$(realpath -e "$ORCA_EXECUTABLE")"
ORCA_ROOT="$(dirname "$ORCA_EXECUTABLE")"
if [ "$ORCA_EXECUTABLE" = "/usr/bin/orca" ]; then
  echo "/usr/bin/orca is Ubuntu's screen reader, not quantum-chemistry ORCA." >&2
  exit 2
fi
if [ ! -x "$OPENMPI_ROOT/bin/mpirun" ] || [ ! -x "$OPENMPI_ROOT/bin/ompi_info" ]; then
  echo "OpenMPI runtime is incomplete under: $OPENMPI_ROOT" >&2
  exit 2
fi
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
  echo "PYTHON_EXECUTABLE is not executable: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

# Do not inherit Intel MPI from the login environment. ORCA 6.0.0's
# shared_openmpi416 build must resolve the matching OpenMPI 4.1.6 runtime.
export PATH="$ORCA_ROOT:$OPENMPI_ROOT/bin:/usr/openpbs/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$OPENMPI_ROOT/lib:$OPENMPI_ROOT/lib64:/usr/openpbs/lib:$ORCA_ROOT:$ORCA_ROOT/lib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY WAYLAND_DISPLAY

MPI_VERSION_OUTPUT="$($OPENMPI_ROOT/bin/mpirun --version)"
if [[ "$MPI_VERSION_OUTPUT" != *"4.1.6"* ]]; then
  echo "Expected OpenMPI 4.1.6, found:" >&2
  printf '%s\n' "$MPI_VERSION_OUTPUT" >&2
  exit 2
fi
if ! "$OPENMPI_ROOT/bin/ompi_info" --parsable --param ras tm | grep -q 'mca:ras:tm'; then
  echo "OpenMPI 4.1.6 lacks the OpenPBS/TM resource-manager component." >&2
  exit 2
fi
if ldd "$ORCA_EXECUTABLE" | grep -q 'not found'; then
  echo "ORCA has unresolved shared-library dependencies:" >&2
  ldd "$ORCA_EXECUTABLE" >&2
  exit 2
fi

MANIFEST_NPROCS="$($PYTHON_EXECUTABLE -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["orca"]["nprocs"]))' "$CONFIG")"
MAXCORE_MB="$($PYTHON_EXECUTABLE -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["orca"]["maxcore_mb_per_process"]))' "$CONFIG")"
ALLOCATED_NCPUS="${PBS_NCPUS:-}"
if [ -z "$ALLOCATED_NCPUS" ] && [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
  ALLOCATED_NCPUS="$(wc -l < "$PBS_NODEFILE")"
fi
if [ -z "$ALLOCATED_NCPUS" ]; then
  echo "Cannot determine the PBS CPU allocation from PBS_NCPUS or PBS_NODEFILE." >&2
  exit 2
fi
if [ "$MANIFEST_NPROCS" -gt "$ALLOCATED_NCPUS" ]; then
  echo "Manifest requests $MANIFEST_NPROCS processes but PBS allocated $ALLOCATED_NCPUS CPUs." >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"

# Ujilab1/3 mount /home over NFS, while /tmp is node-local. Keep all ORCA
# working files local and copy only durable results back to the shared home.
SCRATCH_PARENT="${ORCA_SCRATCH_PARENT:-/tmp}"
SCRATCH_PARENT="$(realpath -e "$SCRATCH_PARENT")"
case "$SCRATCH_PARENT" in
  /tmp|/tmp/*) ;;
  *)
    echo "Refusing non-local or unexpected scratch parent: $SCRATCH_PARENT" >&2
    exit 2
    ;;
esac
SCRATCH_DIR="$(mktemp -d "$SCRATCH_PARENT/uma-orca-${PBS_JOBID:?}-XXXXXX")"
if [ -L "$SCRATCH_DIR" ] || [ ! -d "$SCRATCH_DIR" ]; then
  echo "Scratch creation did not produce a normal directory: $SCRATCH_DIR" >&2
  exit 2
fi
case "$SCRATCH_DIR" in
  "$SCRATCH_PARENT"/uma-orca-"$PBS_JOBID"-*) ;;
  *)
    echo "Unexpected scratch directory: $SCRATCH_DIR" >&2
    exit 2
    ;;
esac

collect_outputs() {
  local candidate
  shopt -s nullglob
  for candidate in \
    "$SCRATCH_DIR"/input.inp \
    "$SCRATCH_DIR"/input.out \
    "$SCRATCH_DIR"/input.engrad \
    "$SCRATCH_DIR"/input.gbw \
    "$SCRATCH_DIR"/input.xyz \
    "$SCRATCH_DIR"/input_property.txt \
    "$SCRATCH_DIR"/input.property.txt; do
    if [ -f "$candidate" ]; then
      cp -p "$candidate" "$RUN_DIR/"
    fi
  done
}

cleanup() {
  local status=$?
  set +e
  collect_outputs
  if [ "$KEEP_SCRATCH" = "1" ]; then
    echo "KEEP_SCRATCH=1; retained $SCRATCH_DIR"
  else
    case "$SCRATCH_DIR" in
      "$SCRATCH_PARENT"/uma-orca-"$PBS_JOBID"-*) rm -rf -- "$SCRATCH_DIR" ;;
      *) echo "Safety check blocked scratch cleanup: $SCRATCH_DIR" >&2 ;;
    esac
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "host=$(hostname)"
echo "job_id=$PBS_JOBID"
echo "allocated_ncpus=$ALLOCATED_NCPUS"
echo "manifest_nprocs=$MANIFEST_NPROCS"
echo "maxcore_mb_per_process=$MAXCORE_MB"
echo "maxcore_budget_mb=$((MANIFEST_NPROCS * MAXCORE_MB))"
echo "orca=$ORCA_EXECUTABLE"
echo "openmpi=$OPENMPI_ROOT"
echo "scratch=$SCRATCH_DIR"
"$OPENMPI_ROOT/bin/mpirun" --version | head -n 3

"$PYTHON_EXECUTABLE" \
  validation/orca_gpu4pyscf/prepare_orca.py \
  "$CONFIG" \
  --output "$SCRATCH_DIR/input.inp"

cd "$SCRATCH_DIR"
# The serial ORCA driver reads %pal and internally starts parallel modules via
# OpenMPI. Never replace this with `mpirun -np N orca ...`.
"$ORCA_EXECUTABLE" input.inp "--bind-to core" > input.out

cd "$WORK_ROOT"
"$PYTHON_EXECUTABLE" \
  validation/orca_gpu4pyscf/parse_orca.py \
  "$CONFIG" \
  "$SCRATCH_DIR/input.engrad" \
  --orca-output "$SCRATCH_DIR/input.out" \
  --output "$RUN_DIR/result.json"

collect_outputs
echo "normalized_result=$RUN_DIR/result.json"
