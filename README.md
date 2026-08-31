# uma_pyscf

This project establishes a reproducible way to fine-tune UMA's `omol`
checkpoints on GPU4PySCF ωB97M-V/def2-TZVPD energy and force labels for
H/Si/Ge/Cl gas-phase molecules, with charge and spin multiplicity stated
explicitly on every structure. Work is split into two parts: **Part I** is the
ORCA / CPU PySCF / GPU4PySCF cross-code validation that decides whether
GPU4PySCF can serve as the teacher-label engine (Gate 1), and **Part II** is the
production library that generates, quality-checks, and trains on those labels.
Gate 1 concluded with a Conditional GO. Part II now provides deterministic
sampling, the canonical label schema, a resumable GPU4PySCF label pipeline,
engineering QC, and leakage-safe splits. The one-structure SoftBank Slurm
smoke and the 50-structure engineering run completed successfully. Dataset
release remains fail closed. A train-only atomic composition baseline and a
checksum-bound non-default state registry are implemented; scientific QC
thresholds and every non-default state still require approval.

## Install

```bash
pip install -e .          # runtime core has no third-party dependencies
pip install -e '.[dev]'   # adds ruff and mypy
uma-pyscf info            # package version, Python version, platform
uma-pyscf label --help    # production protocol dry-run and execution
uma-pyscf import-trajectory --help  # ASE trajectory -> unlabeled candidates
uma-pyscf assemble-portfolio --help # blind source quotas -> calibration manifest
uma-pyscf predict-uma --help  # unlabeled candidate-manifest inference
uma-pyscf select --help       # deterministic, parent-capped acquisition
```

Python 3.10 or newer is required. The lower bound matches the frozen
GPU4PySCF container used by the production label pipeline.

## Test

```bash
python3 -m unittest discover -s tests    # package unit tests, from the repo root
```

`tests/unit/` runs anywhere with the standard library alone. Tests that need
pyscf, ase, or fairchem will live in `tests/integration/` and skip when those
packages are absent.

The Part I validation experiment carries its own test suite:

```bash
cd validation/orca_gpu4pyscf && python3 -m unittest discover -s tests
```

## Layout

```text
src/uma_pyscf/    library: core, schemas, sampling, calculators, QC, datasets, CLI
tests/            unit/ mirrors src/; integration/ and fixtures/ follow later
docs/             project plan, roadmap, milestone plans, lab notes
validation/       Part I cross-code experiment (frozen, self-contained)
```

`validation/orca_gpu4pyscf/` is a **frozen experiment, not part of the
package**. It is never imported from `src/`, and `src/` is never imported from
it. Logic that generalizes is re-implemented with tests on the package side
rather than copied.

`validation/matlantis_pfp/` is a separate multi-fidelity comparison
experiment. It evaluates neutral candidate geometries with a pinned PFP model
inside Matlantis and compares energy differences and forces with canonical
GPU4PySCF records. PFP outputs are not canonical teacher labels and are never
mixed directly into the GPU4PySCF dataset. The production side accepts
versioned unlabeled UMA predictions and acquisition-score records, then routes
selected candidates through the normal GPU4PySCF label and QC path.

Existing molecular ASE trajectories can be imported with source-file hashes,
original frame indices, geometry QC, and duplicate removal before prediction.
Each source can use index-uniform or mass-weighted Cartesian arc-length thinning:

```bash
uma-pyscf import-trajectory \
  configs/sampling/mf_neb_arrhenius_trajectory_pool_v1.yaml \
  --source-root /path/to/neb_arrhenius \
  --output-dir validation/matlantis_pfp/runs/mf_neb_arrhenius_trajectory_pool_v1
```

Selection configs may set both `max_per_parent` and `max_per_trajectory`; the
latter fails closed if any score record lacks trajectory provenance.

Calibration and oracle pools use `uma-pyscf assemble-portfolio`. It reads
multiple immutable candidate manifests, checks their SHA-256 digests, and fills
fixed source quotas without consulting UMA/PFP scores. Cross-source geometry
deduplication includes charge and multiplicity, and parent/trajectory caps stop
one family from dominating the teacher-label budget.

## Documents

- [Project plan](docs/project_plan.md) — scope, gates, operating principles
- [Roadmap](docs/roadmap.md) — current position and next milestone
- [Plans](docs/plans/) — Part I validation, Part II implementation, and the
  production repository structure design that governs this package layout
- [Lab notes](docs/lab_notes/) — dated records of individual investigations
- [P2.3 SoftBank operation](docs/operations/p2_3_softbank_label_pipeline.md) —
  dry-run, Slurm smoke, artifacts, and resume rules
