# Cross-code validation protocol v0.1

## Purpose

Establish whether GPU4PySCF can generate energy and force labels sufficiently
consistent with the ORCA 6.0.0 ωB97M-V/def2-TZVPD level used for the OMol25 task.
The protocol must distinguish GPU implementation differences from genuine
cross-code differences.

## Non-negotiable invariants

- Use identical Cartesian coordinates; no code may optimize the structure.
- Record total charge and spin multiplicity explicitly for every structure.
- Convert multiplicity to PySCF spin as `spin_2s = multiplicity - 1`.
- Reject electron-count/spin-parity mismatches before launching a calculation.
- Record energy in hartree and Cartesian gradient in hartree/bohr.
- A force label is `-gradient`; never compare a force array to a gradient array.
- Record PySCF, LibXC, GPU4PySCF, ORCA, CUDA, and driver versions when present.
- Keep raw outputs under `runs/`; do not commit them.

## Why three engines are required

| Comparison | Primary diagnostic |
|---|---|
| CPU PySCF vs GPU4PySCF | GPU port, precision, density-fitting, and implementation effects |
| CPU PySCF vs ORCA | functional, basis, grids, SCF, RI/COSX, and code-default effects |
| GPU4PySCF vs ORCA | end-to-end suitability of the teacher-data path |

If CPU and GPU PySCF disagree, do not interpret the ORCA comparison yet. If
CPU and GPU agree but both disagree with ORCA, investigate cross-code settings.

## Phase 0: plumbing without expensive DFT

Use H2 with a small HF basis to validate:

- input generation;
- `.engrad` parsing;
- atom order and coordinate units;
- charge/multiplicity conversion;
- gradient sign and shape;
- comparison report and non-zero exit on a failed gate.

This phase validates software plumbing only and says nothing about OMol25
compatibility.

## Phase 1: target-method smoke test

Use the versioned H2 manifest at ωB97M-V/def2-TZVPD. Run CPU PySCF first, then
GPU4PySCF, then ORCA. Require converged SCF in all engines.

PySCF settings:

- `RKS` for `spin_2s == 0`, otherwise `UKS`;
- both `grids.level` and `nlcgrids.level` explicitly set;
- tight SCF convergence;
- analytic gradient with `grid_response = True`;
- no separate D3/D4 correction, because ωB97M-V already contains VV10.

ORCA settings for the approximation-free diagnostic lane:

- `WB97M-V def2-TZVPD EnGrad`;
- `SCNL` to make the nonlocal correlation self-consistent;
- `VeryTightSCF DEFGRID3`;
- `NORI NOCOSX` to remove RI/COSX from the first cross-code diagnostic;
- `NoAutoStart` to prevent an old orbital file from changing the initial state.

The approximation-free ORCA lane is a diagnostic reference, not the OMol25
production recipe. The paper reports RI-J/COSX, DEFGRID3, tight convergence,
`thresh=1e-12`, and `tcut=1e-13`; a released raw `orca.inp` must still be used
to verify the complete input, including nonlocal-correlation settings, before
claiming reproduction.

## Phase 2: numerical convergence ladders

Change one axis at a time.

### PySCF

- ordinary grid levels 4, 5, and 6;
- VV10 `nlcgrids` levels 3, 4, and 5;
- direct SCF versus density fitting, if density fitting is planned for data generation;
- CPU versus GPU at each production candidate setting;
- analytic gradient versus central finite difference for at least one case.

### ORCA

- DEFGRID2 versus DEFGRID3;
- default RIJCOSX versus `NORI NOCOSX`;
- documented nonlocal-correlation choices, including the intended OMol25 setting;
- SCF convergence and integral thresholds.

PySCF and ORCA grid names are not equivalent. Compare convergence within each
code; do not equate PySCF level 5 with ORCA DEFGRID3 by name.

## Phase 3: chemical test matrix

Add geometries incrementally, keeping each charge/spin state as its own case.

| Class | Initial case | Purpose |
|---|---|---|
| closed-shell neutral | H2O singlet | baseline molecular DFT |
| open-shell neutral | CH3 doublet | unrestricted DFT and `<S²>` |
| triplet | O2 triplet | multiplicity and spin contamination |
| cation | H2O+ doublet | charge handling |
| anion | OH− singlet | diffuse basis and charged density |
| same geometry, multiple states | O2 singlet/triplet | state conditioning and state separation |

After light-element cases pass, add only elements relevant to the intended UMA
fine-tuning domain. Heavy elements and ECPs need a separate basis/ECP audit.

## Metrics

For each pair record:

- signed and absolute total-energy difference;
- gradient component RMS difference;
- maximum absolute gradient-component difference and its atom/axis;
- gradient norm per engine;
- `<S²>`, target `S(S+1)`, and deviation when available;
- SCF convergence and wall time.

Absolute cross-code energy is a useful diagnostic but relative energies and
forces are more directly relevant to fine-tuning. Later phases should compare
energy differences within matched chemical transformations and multiple points
on the same potential-energy surface.

## Acceptance policy

The JSON manifest contains provisional tolerances so automation can fail
closed. Do not freeze scientific acceptance thresholds from H2 alone. Final
thresholds require:

1. CPU/GPU PySCF numerical equivalence across the test matrix;
2. within-code grid convergence tighter than the cross-code discrepancy;
3. stable conclusions with and without ORCA RIJCOSX;
4. explicit confirmation of the complete OMol25 ORCA input from a released raw case;
5. evaluation in the energy/force units and error scales used for UMA training.

## Execution policy

- CPU PySCF runs on a PBS compute node reached through `ssh ujilab`, not on the login node.
- GPU4PySCF runs on an NVIDIA GPU node with a pinned CUDA/CuPy/cuTENSOR stack.
- ORCA runs on CPU cores, preferably with local scratch and `%pal nprocs`.
- The main ORCA driver is not launched with `mpirun`.
- Inputs and normalized small summaries may be copied between machines; raw outputs stay remote by default.
