# uma_pyscf

This project establishes a reproducible way to fine-tune UMA's `omol`
checkpoints on GPU4PySCF ωB97M-V/def2-TZVPD energy and force labels for
H/Si/Ge/Cl gas-phase molecules, with charge and spin multiplicity stated
explicitly on every structure. Work is split into two parts: **Part I** is the
ORCA / CPU PySCF / GPU4PySCF cross-code validation that decides whether
GPU4PySCF can serve as the teacher-label engine (Gate 1), and **Part II** is the
production library that generates, quality-checks, and trains on those labels.
Part II is at its foundation milestone: the package currently provides the
cross-cutting `core` primitives (units, spin, identifiers, atomic I/O, error
hierarchy) and the `uma-pyscf` CLI skeleton.

## Install

```bash
pip install -e .          # runtime core has no third-party dependencies
pip install -e '.[dev]'   # adds ruff and mypy
uma-pyscf info            # package version, Python version, platform
```

Python 3.11 or newer is required.

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
src/uma_pyscf/    library: core/ (units, spin, ids, io, errors), cli/
tests/            unit/ mirrors src/; integration/ and fixtures/ follow later
docs/             project plan, roadmap, milestone plans, lab notes
validation/       Part I cross-code experiment (frozen, self-contained)
```

`validation/orca_gpu4pyscf/` is a **frozen experiment, not part of the
package**. It is never imported from `src/`, and `src/` is never imported from
it. Logic that generalizes is re-implemented with tests on the package side
rather than copied.

## Documents

- [Project plan](docs/project_plan.md) — scope, gates, operating principles
- [Roadmap](docs/roadmap.md) — current position and next milestone
- [Plans](docs/plans/) — Part I validation, Part II implementation, and the
  production repository structure design that governs this package layout
- [Lab notes](docs/lab_notes/) — dated records of individual investigations
