#!/usr/bin/env bash
# Verify OpenMPI 4.1.6 and OpenPBS/TM integration before installing ORCA.
#PBS -N test_ompi416
#PBS -q workq
#PBS -j oe
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=1gb

set -euo pipefail

OPENMPI_ROOT="${OPENMPI_ROOT:-/home/seki/uma_pyscf/software/openmpi/4.1.6}"
export PATH="$OPENMPI_ROOT/bin:/usr/openpbs/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$OPENMPI_ROOT/lib:$OPENMPI_ROOT/lib64:/usr/openpbs/lib"
unset DISPLAY WAYLAND_DISPLAY

test -x "$OPENMPI_ROOT/bin/mpirun"
test -x "$OPENMPI_ROOT/bin/ompi_info"

echo "job_id=${PBS_JOBID:?}"
ALLOCATED_NCPUS="${PBS_NCPUS:-}"
if [ -z "$ALLOCATED_NCPUS" ] && [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
  ALLOCATED_NCPUS="$(wc -l < "$PBS_NODEFILE")"
fi
if [ "$ALLOCATED_NCPUS" != "4" ]; then
  echo "Expected four allocated CPUs, found: ${ALLOCATED_NCPUS:-unknown}" >&2
  exit 2
fi
echo "allocated_ncpus=$ALLOCATED_NCPUS"
echo "pbs_nodefile=${PBS_NODEFILE:-unset}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
  sort "$PBS_NODEFILE" | uniq -c
fi
echo "execution_host=$(hostname)"
"$OPENMPI_ROOT/bin/mpirun" --version
"$OPENMPI_ROOT/bin/ompi_info" --parsable --param ras tm

"$OPENMPI_ROOT/bin/mpirun" -np 4 --bind-to core --report-bindings \
  /bin/bash -c 'printf "rank=%s host=%s cpu=%s\n" "$OMPI_COMM_WORLD_RANK" "$(hostname)" "$(taskset -pc $$ 2>/dev/null | sed "s/.*: //")"'
