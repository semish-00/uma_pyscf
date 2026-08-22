# ORCA 6.0.0 setup on ujilab

This guide keeps the licensed ORCA distribution outside Git and installs the
matching OpenMPI runtime in a versioned user prefix. It is intentionally
specific to the environment observed on ujilab on 2026-08-13.

## Target layout

```text
/home/seki/uma_pyscf/software/orca/6.0.0/       licensed ORCA files
/home/seki/uma_pyscf/software/openmpi/4.1.6/    ORCA-compatible MPI runtime
/home/seki/uma_pyscf/installers/                private, untracked archives
```

Do not place the ORCA installer, extracted binaries, license-related files, or
raw calculation outputs in this repository. `/usr/bin/orca` on ujilab is the
Ubuntu screen reader and must never be used for quantum chemistry.

## 1. Obtain ORCA 6.0.0 legitimately

OMol25 used ORCA 6.0.0. After confirming that the ORCA terms cover this
project and its teacher-data use, obtain the registered-user Linux x86-64
OpenMPI 4.1.6 build. The expected installer name is of the form:

```text
orca_6_0_0_linux_x86-64_shared_openmpi416.run
```

Transfer it to a private, untracked location on ujilab. Run the installer with
a versioned destination rather than accepting an ambiguous default, following
the installer's own help and the official `-- -p <path>` mechanism:

```bash
chmod 700 /home/seki/uma_pyscf/installers/orca_6_0_0_linux_x86-64_shared_openmpi416.run
/home/seki/uma_pyscf/installers/orca_6_0_0_linux_x86-64_shared_openmpi416.run \
  -- -p /home/seki/uma_pyscf/software/orca/6.0.0
```

The ORCA installer may update a shell startup file. PBS jobs in this project do
not rely on that change: they set `ORCA_ROOT`, `PATH`, and `LD_LIBRARY_PATH`
explicitly to make runs reproducible and to prevent Intel MPI from leaking in.

If exact 6.0.0 is no longer available to the account, do not silently replace
it. Install the available version under a different prefix and label it as a
diagnostic lane; it cannot establish exact OMol25 reproduction by itself.

## 2. Build OpenMPI 4.1.6 through PBS

Do not use the installed Intel MPI 2021.10 and do not substitute a conda MPI.
The ORCA distribution is dynamically linked for OpenMPI 4.1.6.

Download `openmpi-4.1.6.tar.gz` from the official OpenMPI release page. Its
published SHA-256 is:

```text
44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566
```

Place the archive outside Git, then build it on a compute node so compilation
does not load the login node:

```bash
/usr/openpbs/bin/qsub \
  -v OPENMPI_TARBALL=/home/seki/uma_pyscf/installers/openmpi-4.1.6.tar.gz \
  validation/orca_gpu4pyscf/jobs/build_openmpi_416_pbs.sh
```

The build script:

- verifies the exact upstream SHA-256;
- uses GCC/G++/GFortran 9.4.0 already present on ujilab;
- configures `--with-tm=/usr/openpbs` using the installed OpenPBS headers and
  libraries so OpenMPI can honor PBS allocations;
- builds in node-local `/tmp`;
- retains the default Fortran support required by the ORCA documentation;
- installs once to `/home/seki/uma_pyscf/software/openmpi/4.1.6` on shared NFS;
- refuses to overwrite a non-empty prefix.

After the build finishes, verify four ranks inside a PBS allocation:

```bash
/usr/openpbs/bin/qsub \
  validation/orca_gpu4pyscf/jobs/test_openmpi_416_pbs.sh
```

## 3. Preflight

Before the first calculation, verify the resolved programs explicitly:

```bash
/home/seki/uma_pyscf/software/openmpi/4.1.6/bin/mpirun --version
/home/seki/uma_pyscf/software/openmpi/4.1.6/bin/ompi_info --version
/home/seki/uma_pyscf/software/openmpi/4.1.6/bin/ompi_info --parsable --param ras tm
ldd /home/seki/uma_pyscf/software/orca/6.0.0/orca
```

Expected MPI version is exactly `4.1.6`, the `ras:tm` component must be listed,
and `ldd` must not contain `not found`. The PBS runner repeats these checks and
fails closed.

## 4. Submit the H2 smoke calculation

From the repository root on ujilab:

```bash
/usr/openpbs/bin/qsub \
  validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh
```

The manifest requests four ORCA processes and `%maxcore 2000` MB per process.
The PBS template allocates four CPU cores and 16 GB total, leaving ample memory
outside the 8 GB ORCA MaxCore budget.

The job invokes:

```bash
/home/seki/uma_pyscf/software/orca/6.0.0/orca input.inp "--bind-to core"
```

It deliberately does **not** invoke:

```bash
mpirun -np 4 /home/seki/uma_pyscf/software/orca/6.0.0/orca input.inp
```

The first command starts the serial ORCA driver. The `%pal nprocs 4` block in
`input.inp` instructs that driver to launch the parallel ORCA modules using the
OpenMPI runtime. The second command incorrectly launches multiple copies of the
driver.

## 5. PBS resource mapping

For a normal single-node job, these quantities must agree:

| Layer | Four-process smoke value |
|---|---:|
| PBS `select=1:ncpus=...:mpiprocs=...` | 4 |
| ORCA `%pal nprocs` | 4 |
| OpenMP/BLAS thread count per process | 1 |
| `%maxcore` | 2000 MB/process |
| Approximate MaxCore budget | 8 GB |
| PBS memory | 16 GB |

The runner checks that `%pal nprocs` does not exceed the PBS allocation and forces
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` to prevent accidental
oversubscription. `%maxcore` is per process, not per job. Keep
`nprocs × maxcore` comfortably below the PBS memory request; using at most
about 70–75% is a conservative project rule, not an exact cap on every ORCA
allocation.

On the current ujilab OpenPBS installation, `PBS_NCPUS` is not necessarily
exported into the job environment. The runner therefore derives the allocation
from the number of entries in `PBS_NODEFILE` when needed.

## 6. Scratch and durable outputs

On Ujilab1 and Ujilab3:

- `/home` is shared NFS;
- `/tmp` is node-local ext4 with about 741 GB free at inspection time.

The runner therefore creates `/tmp/uma-orca-$PBS_JOBID-*`, runs entirely there,
copies back the input, output, gradient, wavefunction, and property files that
exist, and removes the temporary directory after validation. Set
`KEEP_SCRATCH=1` only for a failed-run diagnosis; clean retained directories
promptly.

## 7. Scaling policy

Keep the first validation jobs on one node. No PBS nodefile is needed for a
single-node ORCA calculation, and avoiding multi-node execution removes SSH,
network, and distributed-scratch variables from the cross-code comparison.

Recommended first measurements:

1. H2 at 1 and 4 processes for correctness;
2. H2O or another modest molecule at 4, 8, and 16 processes;
3. a representative 20–60 atom case at 8 and 16 processes;
4. adopt the smallest process count near the measured performance plateau.

ORCA's manual says RI-DFT commonly benefits up to about 16 processes and that
parallel overhead becomes important beyond that, while hybrid DFT may benefit
from a few more. Do not allocate all 64 cores merely because a node has them.

## References

- [ORCA 6.0 serial and parallel execution](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)
- [ORCA 6.0 energy, gradient, and per-process memory](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/energygradients.html)
- [ORCA 6.0 installer example](https://www.faccts.de/docs/orca/6.0/manual/_downloads/f5a34d500b44971b2b057d96d7f899ca/orca.pdf)
- [OpenMPI 4.1 official release archive and checksums](https://www.open-mpi.org/software/ompi/v4.1/)
- [OpenMPI runtime-system configure options (`--with-tm`)](https://docs.open-mpi.org/en/main/installing-open-mpi/configure-cli-options/runtime.html)
- [OMol25 calculation details](https://arxiv.org/html/2505.08762)
