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
configs/                versioned calculation manifests
structures/             small, versioned XYZ inputs
suites/                 versioned case collections (29-case ladder, GPU smoke,
                        charge/spin mini matrix)
jobs/                   PBS templates for ujilab CPU calculations
setup/                  versioned installation and environment guidance
tests/                  parser and validation tests without ORCA/PySCF
runs/                   generated inputs and results; ignored by Git
common.py               manifest, XYZ, and charge/spin validation
run_pyscf.py            CPU PySCF or GPU4PySCF energy/gradient runner
run_suite.py            sequential non-PBS suite runner with attempt ledger
submit_suite.py         PBS submission of a suite on ujilab
generate_ladder_suite.py  deterministic 29-case ladder generator
generate_charge_spin_mini_suite.py  deterministic charge/spin mini matrix
collect_environment.py  Workstream A1 host inventory (GPU/driver/CUDA/packages)
gpu_smoke_check.py      Workstream A2 installation smoke test
prepare_orca.py         deterministic ORCA input generator
parse_orca.py           ORCA .engrad normalizer
compare.py              normalized-result comparison and acceptance report
summarize_suite.py      suite-level comparison summary for any engine pair
export_parity_csv.py    parity-plot CSV export for any engine pair
gate1_metrics.py        Gate 1 metric tables and provisional CPU–GPU gate
protocol.md             scientific comparison protocol and staged test set
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

On the GPU host, first record the environment and verify the installed stack.
Both commands are batch-safe and need no login shell:

```bash
python collect_environment.py   # writes configs/environments/gpu4pyscf-<host>.yaml
python gpu_smoke_check.py --output runs/gpu_smoke_check.json
```

`gpu_smoke_check.py` walks the stack in order — CuPy import, GPU visibility,
kernel launch, PySCF/GPU4PySCF import, then a tiny ωB97M-V/def2-TZVPD RKS and
UKS energy plus analytic gradient on the GPU — and stops at the first broken
layer so driver/CUDA/CuPy/GPU4PySCF boundaries stay separable.

Run a single manifest in a compatible CUDA environment:

```bash
python run_pyscf.py configs/h2_wb97mv_def2tzvpd.json \
  --device gpu \
  --output runs/h2_wb97mv_def2tzvpd/gpu4pyscf/result.json
```

The GPU runner intentionally starts from a normal PySCF DFT object and calls
`to_gpu()`. Both the ordinary DFT grid and the separate VV10 nonlocal grid are
set before conversion, and the runner refuses to continue if the conversion
changed either grid level.

Suites run sequentially without PBS through `run_suite.py`. Each case runs in
its own child process; results are written atomically on success only, and
every attempt is appended to `runs/<case>/gpu4pyscf/attempts.jsonl`, which is
never overwritten. The five-case C1 smoke ladder stops at the first failure by
design — do not start the 29-case ladder until it passes:

```bash
python run_suite.py suites/gpu_smoke_v1.json --device gpu --dry-run   # validate only
python run_suite.py suites/gpu_smoke_v1.json --device gpu
python run_suite.py suites/si_ge_h_cl_ladder_v1.json --device gpu
```

### Charge/spin mini matrix

`suites/charge_spin_mini_v1.json` is the plan's additional scope: SiH3 and GeH3
held at their existing neutral doublet seed geometries in five further
charge/multiplicity states each, twelve cases in total. Because the geometry is
fixed, the suite isolates CPU–GPU numerical parity across charge and spin
states. `generate_charge_spin_mini_suite.py` regenerates it deterministically
and creates no structure file.

Its `state_selection_status` is `pending_scientific_review`. These states are
numerical-parity probes, not approved training labels; do not promote their
results to teacher data before the scientific state selection is reviewed. Run
it only after the five-case smoke suite passes:

```bash
python generate_charge_spin_mini_suite.py                                  # regenerate
python run_suite.py suites/charge_spin_mini_v1.json --device gpu --dry-run
python run_suite.py suites/charge_spin_mini_v1.json --device cpu
python run_suite.py suites/charge_spin_mini_v1.json --device gpu
```

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

Suite-level summaries and parity CSVs accept any engine pair. The Part I
priority comparison is GPU4PySCF against CPU PySCF:

```bash
python summarize_suite.py suites/si_ge_h_cl_ladder_v1.json \
  --root ../.. \
  --left-engine gpu4pyscf --right-engine pyscf-cpu \
  --write-comparisons --output runs/gpu_vs_cpu_summary.json

python export_parity_csv.py --left-engine gpu4pyscf --right-engine pyscf-cpu
```

`gate1_metrics.py` aggregates the Gate 1 metric table for a whole suite across
all three engine pairs at once and evaluates the provisional CPU–GPU numeric
gate:

```bash
python gate1_metrics.py suites/si_ge_h_cl_ladder_v1.json
```

It writes three files into `analysis/`: `gate1_case_metrics_<suite>.csv`, one
row per case and engine pair with the signed and absolute energy difference,
the gradient component RMSE/MAE/max, and where the max sits;
`gate1_performance_<suite>.csv`, one row per case and engine with `<S^2>`, wall
time, and the SCF/gradient split where a result records it; and
`gate1_summary_<suite>.json`, holding the worst offender per pair, the per-case
verdict against the provisional 5e-6 Eh, 2e-5 Eh/bohr, and 1e-4 Eh/bohr
thresholds, the "CPU–GPU difference stays below the ORCA–CPU difference"
relative check, and the missing results per engine. Two results are only
compared when their case id, input fingerprint, and convergence agree;
anything else raises rather than reporting a difference that is really an input
difference. The tool always exits 0 — it reports, it does not enforce.

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
