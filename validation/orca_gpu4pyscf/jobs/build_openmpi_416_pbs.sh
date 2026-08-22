#!/usr/bin/env bash
# Build the ORCA-compatible OpenMPI 4.1.6 runtime on an ujilab compute node.
# The caller supplies a previously downloaded, checksum-verified source tarball.
#
# Example:
#   /usr/openpbs/bin/qsub \
#     -v OPENMPI_TARBALL=/home/seki/uma_pyscf/installers/openmpi-4.1.6.tar.gz \
#     validation/orca_gpu4pyscf/jobs/build_openmpi_416_pbs.sh
#PBS -N build_ompi416
#PBS -q workq
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=8gb

set -euo pipefail

OPENMPI_TARBALL="${OPENMPI_TARBALL:?Set OPENMPI_TARBALL to openmpi-4.1.6.tar.gz}"
OPENMPI_PREFIX="${OPENMPI_PREFIX:-/home/seki/uma_pyscf/software/openmpi/4.1.6}"
EXPECTED_SHA256="44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566"

if [ ! -f "$OPENMPI_TARBALL" ]; then
  echo "Tarball does not exist: $OPENMPI_TARBALL" >&2
  exit 2
fi
case "$(basename "$OPENMPI_TARBALL")" in
  openmpi-4.1.6.tar.gz) ;;
  *) echo "Expected openmpi-4.1.6.tar.gz, got: $OPENMPI_TARBALL" >&2; exit 2 ;;
esac
ACTUAL_SHA256="$(sha256sum "$OPENMPI_TARBALL" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "OpenMPI source checksum mismatch: $ACTUAL_SHA256" >&2
  exit 2
fi
if [ -e "$OPENMPI_PREFIX" ] && [ -n "$(find "$OPENMPI_PREFIX" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "Refusing to overwrite non-empty prefix: $OPENMPI_PREFIX" >&2
  exit 2
fi

BUILD_ROOT="$(mktemp -d "/tmp/openmpi-4.1.6-${PBS_JOBID:?}-XXXXXX")"
case "$BUILD_ROOT" in
  /tmp/openmpi-4.1.6-"$PBS_JOBID"-*) ;;
  *) echo "Unexpected build directory: $BUILD_ROOT" >&2; exit 2 ;;
esac
cleanup() {
  local status=$?
  case "$BUILD_ROOT" in
    /tmp/openmpi-4.1.6-"$PBS_JOBID"-*) rm -rf -- "$BUILD_ROOT" ;;
    *) echo "Safety check blocked build-directory cleanup: $BUILD_ROOT" >&2 ;;
  esac
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OPENMPI_PREFIX"
tar -xzf "$OPENMPI_TARBALL" -C "$BUILD_ROOT" --strip-components=1
cd "$BUILD_ROOT"

./configure \
  --prefix="$OPENMPI_PREFIX" \
  --with-tm=/usr/openpbs \
  CC=/usr/bin/gcc \
  CXX=/usr/bin/g++ \
  FC=/usr/bin/gfortran
make -j "${PBS_NCPUS:-8}"
make install

"$OPENMPI_PREFIX/bin/mpirun" --version
"$OPENMPI_PREFIX/bin/ompi_info" --version
if ! "$OPENMPI_PREFIX/bin/ompi_info" --parsable --param ras tm | grep -q 'mca:ras:tm'; then
  echo "OpenMPI was built without the requested OpenPBS/TM support." >&2
  exit 2
fi
echo "installed_openmpi=$OPENMPI_PREFIX"
