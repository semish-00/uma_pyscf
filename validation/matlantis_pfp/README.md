# Matlantis/PFP comparison experiment

This directory evaluates the same neutral, closed-shell geometries used by
`uma_pyscf` with PFP inside Matlantis. It is deliberately separate from
`src/uma_pyscf/` and from canonical GPU4PySCF labels.

## Scientific boundary

- PFP does not expose the total charge and spin multiplicity as selectable
  state inputs. This first protocol therefore accepts only the project's
  neutral-singlet comparison set and does not claim state-resolved parity.
- PFP `v9.0.0` / `R2SCAN_PLUS_D3` is pinned because Si and Ge are outside the
  nine-element `WB97XD` mode. This is a different theory level from the
  GPU4PySCF `omegaB97M-V/def2-TZVPD` teacher.
- Absolute total energies from the two methods must not be interpreted as
  parity metrics. `compare.py` reports same-composition-centered energy errors
  and Cartesian force-component errors.
- PFP records use their own schema. Do not pass them to the canonical
  ASE-LMDB exporter or mix them into the GPU4PySCF teacher dataset.

PFP data can still help as a structure generator, an active-learning proposal
source, or a separately identified lower-fidelity/distillation dataset. Any of
those routes needs a held-out GPU4PySCF validation gate before adoption.

## Layout

```text
validation/matlantis_pfp/
  configs/              pinned PFP protocol
  run_single_points.py  Matlantis-only, resumable evaluator
  run_langevin_md.py    pinned neutral-singlet NVT candidate generator
  compare.py            local PFP vs canonical GPU4PySCF comparison
  build_acquisition_scores.py
                        PFP-versus-UMA scores; never reads HF result fields
  runs/                 generated artifacts; ignored by Git
  tests/                 comparison-unit tests
```

## Run

Generate the deterministic engineering manifest from the repository root:

```bash
python -m uma_pyscf.cli.main sample \
  configs/sampling/engineering_50_v1.yaml \
  --output-dir validation/matlantis_pfp/runs/engineering_50_input
```

On Matlantis, activate an existing kernel and perform a dry-run before the
foreground calculation:

```bash
use_venv python311
python validation/matlantis_pfp/run_single_points.py \
  --config validation/matlantis_pfp/configs/pfp_v9_r2scan_plus_d3_v1.json \
  --manifest validation/matlantis_pfp/runs/engineering_50_input/engineering_50_v1_candidates.json \
  --output-dir validation/matlantis_pfp/runs/engineering_50_pfp_v9_r2scan_plus_d3 \
  --dry-run

python validation/matlantis_pfp/run_single_points.py \
  --config validation/matlantis_pfp/configs/pfp_v9_r2scan_plus_d3_v1.json \
  --manifest validation/matlantis_pfp/runs/engineering_50_input/engineering_50_v1_candidates.json \
  --output-dir validation/matlantis_pfp/runs/engineering_50_pfp_v9_r2scan_plus_d3 \
  --keep-going
```

This small single-point batch is suitable for foreground execution. For a
longer calculation, the Matlantis alternative is a Notebook submitted with
`mtl-bg-job run --kernel <existing-kernel> ...`; keep the repository and all
referenced inputs in the shared Matlantis path.

After retrieving PFP records, compare them with accepted canonical labels:

```bash
python validation/matlantis_pfp/compare.py \
  --pfp-records validation/matlantis_pfp/runs/engineering_50_pfp_v9_r2scan_plus_d3/records \
  --reference-records /path/to/engineering_50/qc/records \
  --split /path/to/engineering_50_baseline_split_v1.json \
  --output-dir validation/matlantis_pfp/runs/engineering_50_comparison
```

## Selection plumbing dry-run

The already observed engineering set can validate score-record and selection
plumbing, but it is not scientific active-learning evidence. Regenerate its
candidate manifest, build scores from the base UMA predictions and PFP records,
then run the production selector without requesting any new GPU4PySCF labels:

```bash
python validation/matlantis_pfp/build_acquisition_scores.py \
  --candidates validation/matlantis_pfp/runs/engineering_50_input/engineering_50_v1_candidates.json \
  --uma-predictions validation/matlantis_pfp/runs/engineering_50_existing_evaluations/base_uma_evaluation.json \
  --pfp-records validation/matlantis_pfp/runs/engineering_50_pfp_v9_r2scan_plus_d3/records \
  --output validation/matlantis_pfp/runs/engineering_50_acquisition_dry_run/scores.json

python -m uma_pyscf.cli.main select \
  --scores validation/matlantis_pfp/runs/engineering_50_acquisition_dry_run/scores.json \
  --config configs/sampling/mf_pfp_screening_engineering_50_dry_run_v1.yaml \
  --output validation/matlantis_pfp/runs/engineering_50_acquisition_dry_run/selection.json
```

## Reaction-trajectory engineering pool

Import the four existing `neb_arrhenius` IRC trajectories without copying
reference energies or forces:

```bash
python -m uma_pyscf.cli.main import-trajectory \
  configs/sampling/mf_neb_arrhenius_trajectory_pool_v1.yaml \
  --source-root /path/to/neb_arrhenius \
  --output-dir validation/matlantis_pfp/runs/mf_neb_arrhenius_trajectory_pool_v1
```

The committed config requests 20 endpoint-inclusive frames from each path. On
the 2026-08-13 source runs this produces 80 proposals: 78 accepted and two
shared transition-state frames rejected as duplicates. This is still an
engineering pool with only two parent reaction families, not a fixed test.

Run PFP on the accepted manifest inside Matlantis, then resume once to verify
that every validated record is skipped:

```bash
use_venv python311
python validation/matlantis_pfp/run_single_points.py \
  --config validation/matlantis_pfp/configs/pfp_v9_r2scan_plus_d3_v1.json \
  --manifest validation/matlantis_pfp/runs/mf_neb_arrhenius_trajectory_pool_v1_input/mf_neb_arrhenius_trajectory_pool_v1_candidates.json \
  --output-dir validation/matlantis_pfp/runs/mf_neb_arrhenius_trajectory_pool_v1_pfp_v9_r2scan_plus_d3 \
  --keep-going
```

The 2026-09-01 run completed 78/78 records with no failures; its resume pass
completed zero and skipped all 78. After base UMA inference, the committed
selection config produced 10 records per policy and a 22-record union while
enforcing five records per parent and three per trajectory.

## PFP Langevin MD candidate source

PFP MD is restricted to neutral singlets and creates candidate trajectories,
not canonical labels. The committed preflight uses PFP v9.0.0 /
`R2SCAN_PLUS_D3`, NVT Langevin, a 0.5 fs timestep, 200 steps, temperatures
300/600/900/1200 K, and two independent seeds. It records the full run grid,
model protocol, seed, diagnostics, trajectory path, and runtime versions.

The five committed seed parents are engineering fixtures shared with
`engineering_50`; they validate the runner only and must not enter C0 or T0:

```bash
python -m uma_pyscf.cli.main sample \
  configs/sampling/pfp_md_engineering_seeds_v1.yaml \
  --output-dir validation/matlantis_pfp/runs/pfp_md_engineering_seeds_v1

use_venv python311
python validation/matlantis_pfp/run_langevin_md.py \
  --config validation/matlantis_pfp/configs/pfp_v9_r2scan_plus_d3_langevin_preflight_v1.json \
  --manifest validation/matlantis_pfp/runs/pfp_md_engineering_seeds_v1/pfp_md_engineering_seeds_v1_candidates.json \
  --output-dir validation/matlantis_pfp/runs/pfp_md_engineering_preflight_v1 \
  --dry-run
```

Remove `--dry-run` only inside Matlantis. Each estimator/calculator is created
per trajectory; velocity and thermostat RNGs are separately seeded. A partial
trajectory without a completed summary is refused on resume instead of being
silently overwritten. After the engineering preflight, C0 uses a new,
independent parent manifest and imports decorrelated frames through
`uma-pyscf import-trajectory` with `mass_weighted_arc_length`.
