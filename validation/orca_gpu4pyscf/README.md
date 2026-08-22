# ORCA–PySCF–GPU4PySCF cross-code validation

This directory contains a self-contained validation experiment. It is not part
of the future `uma_pyscf` library and should not be imported from `src/`.

The central design is a three-way comparison:

1. **CPU PySCF ↔ GPU4PySCF** isolates changes introduced by GPU acceleration.
2. **CPU PySCF ↔ ORCA** isolates differences between quantum-chemistry codes.
3. **GPU4PySCF ↔ ORCA** checks the intended teacher-data path end to end.

All three calculations must use the same Cartesian geometry, charge, spin
multiplicity, functional, basis, and intentionally selected numerical settings.
Geometry optimization is outside the first validation phase.

## Layout

```text
configs/          versioned calculation manifests
structures/       small, versioned XYZ inputs
jobs/             PBS templates for ujilab CPU calculations
setup/            versioned installation and environment guidance
tests/            parser and validation tests without ORCA/PySCF
runs/             generated inputs and results; ignored by Git
common.py         manifest, XYZ, and charge/spin validation
run_pyscf.py      CPU PySCF or GPU4PySCF energy/gradient runner
prepare_orca.py   deterministic ORCA input generator
parse_orca.py     ORCA .engrad normalizer
compare.py        normalized-result comparison and acceptance report
protocol.md       scientific comparison protocol and staged test set
```

## Local dry run

None of these commands imports PySCF or invokes ORCA:

```bash
cd validation/orca_gpu4pyscf
python prepare_orca.py configs/h2_wb97mv_def2tzvpd.json --stdout
python -m unittest discover -s tests -v
```

## CPU PySCF

The production CPU environment is on the PBS cluster reached through
`ssh ujilab`. Do not run a long PySCF calculation on its login node. The
runner itself is:

```bash
python run_pyscf.py configs/h2_wb97mv_def2tzvpd.json \
  --device cpu \
  --output runs/h2_wb97mv_def2tzvpd/pyscf-cpu/result.json
```

The `--dry-run` form validates and prints the resolved calculation without
importing PySCF:

```bash
python run_pyscf.py configs/h2_wb97mv_def2tzvpd.json --device cpu --dry-run
```

After cloning this repository on ujilab, submit from the checkout root:

```bash
qsub -v CONFIG=validation/orca_gpu4pyscf/configs/h2_wb97mv_def2tzvpd.json \
  validation/orca_gpu4pyscf/jobs/run_pyscf_cpu_pbs.sh
```

The template defaults to the already established qcthermo PySCF interpreter,
`/home/seki/miniconda3/envs/qcthermo/bin/python`, but the path remains
overridable through `PYTHON_EXECUTABLE`.

## GPU4PySCF

Run the same manifest in a compatible CUDA environment:

```bash
python run_pyscf.py configs/h2_wb97mv_def2tzvpd.json \
  --device gpu \
  --output runs/h2_wb97mv_def2tzvpd/gpu4pyscf/result.json
```

The GPU runner intentionally starts from a normal PySCF DFT object and calls
`to_gpu()`. Both the ordinary DFT grid and the separate VV10 nonlocal grid are
set before conversion.

## ORCA

Generate an input file, then run ORCA in a node-local scratch directory. ORCA
is a CPU/OpenMPI code for this calculation; do not reserve a GPU only for ORCA.
OMol25 used ORCA 6.0.0, so that version is the reproduction baseline. On
`ujilab`, `/usr/bin/orca` is Ubuntu's screen reader rather than the quantum
chemistry program; always provide the verified quantum-chemistry binary by its
full path.

```bash
python prepare_orca.py configs/h2_wb97mv_def2tzvpd.json \
  --output runs/h2_wb97mv_def2tzvpd/orca/input.inp

/full/path/to/orca runs/h2_wb97mv_def2tzvpd/orca/input.inp \
  > runs/h2_wb97mv_def2tzvpd/orca/input.out

python parse_orca.py \
  configs/h2_wb97mv_def2tzvpd.json \
  runs/h2_wb97mv_def2tzvpd/orca/input.engrad \
  --orca-output runs/h2_wb97mv_def2tzvpd/orca/input.out \
  --output runs/h2_wb97mv_def2tzvpd/orca/result.json
```

Do not prefix the ORCA driver with `mpirun`. ORCA starts its parallel modules
from `%pal nprocs` in the input, and the driver should be called by its full
path.

If ORCA is installed on a PBS host, the equivalent submission is:

```bash
/usr/openpbs/bin/qsub \
  validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh
```

The ujilab-specific installation layout, OpenMPI build job, memory mapping,
and scratch policy are documented in [`setup/ujilab.md`](setup/ujilab.md).

## Compare

```bash
python compare.py \
  runs/h2_wb97mv_def2tzvpd/pyscf-cpu/result.json \
  runs/h2_wb97mv_def2tzvpd/gpu4pyscf/result.json \
  --output runs/h2_wb97mv_def2tzvpd/cpu-vs-gpu.json

python compare.py \
  runs/h2_wb97mv_def2tzvpd/pyscf-cpu/result.json \
  runs/h2_wb97mv_def2tzvpd/orca/result.json \
  --output runs/h2_wb97mv_def2tzvpd/cpu-vs-orca.json
```

Thresholds in the example manifest are provisional engineering gates, not a
claim that different codes must agree to those values. Freeze final thresholds
only after the grid and approximation ladders in `protocol.md` are complete.

## References

- [ORCA 6.0 parallel execution](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)
- [OMol25 calculation details](https://arxiv.org/html/2505.08762)
- [ORCA numerical integration](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalintegration.html)
- [ORCA RI/COSX](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/RI.html)
- [PySCF DFT and VV10](https://pyscf.org/user/dft.html)
- [GPU4PySCF features and installation](https://github.com/pyscf/gpu4pyscf)
