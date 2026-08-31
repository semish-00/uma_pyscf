# Matlantis/PFP engineering-50 single-point comparison

- Date: 2026-08-31
- Matlantis SSH host: `matlantis_aix`
- Remote root: `/home/jovyan/uma_pyscf_matlantis_pilot_20260831`
- PFP: model `v9.0.0`, calc mode `R2SCAN_PLUS_D3`
- Runtime: Python 3.11.11, ASE 3.25.0, pfp-api-client 1.21.3
- Input: `engineering_50_v1`, 50/50 geometry-QC accepted neutral singlets
- Reference: accepted GPU4PySCF `omegaB97M-V/def2-TZVPD` records from job 1797134

## Repository history and experiment boundary

Before this run, the repository contained a feasibility note about PFP's
charge/spin scope but no `pfp_api_client` execution code or PFP result
artifacts. The PFP experiment now lives under `validation/matlantis_pfp/` and
remains separate from the canonical GPU4PySCF label pipeline.

PFP does not expose charge and multiplicity as user-selectable state inputs.
The pilot therefore fails closed unless every candidate is a neutral singlet.
`WB97XD` cannot be used because its supported element set excludes Si and Ge;
the protocol pins PFP v9.0.0 / `R2SCAN_PLUS_D3` instead. This is not the same
theory level as the GPU4PySCF teacher.

## Execution result

The local deterministic sampler reproduced 50/50 accepted geometries. A
Matlantis dry-run validated all 50 candidates, followed by a one-record smoke
and the full foreground single-point batch.

| Pass | Completed | Skipped | Failed | Records present |
|---|---:|---:|---:|---:|
| one-record smoke | 1 | 0 | 0 | 1 |
| full first pass | 50 | 0 | 0 | 50 |
| same-directory resume | 0 | 50 | 0 | 50 |

PFP API wall time stored in the 50 records totaled 2.534 s: mean 0.0507 s,
median 0.0536 s, minimum 0.0348 s, and maximum 0.0654 s per structure. This
excludes SSH startup and Python environment activation. A separate Estimator
and ASECalculator was constructed for every structure.

The recovered records are stored locally under the Git-ignored path:

```text
validation/matlantis_pfp/runs/engineering_50_pfp_v9_r2scan_plus_d3/
```

The canonical digest of the sorted 50 per-record SHA-256 lines is
`cd777ab637d4f977f5476e1f4c01f2b22dc34643391136a2caab1da271246208`.

## PFP versus GPU4PySCF

All 50 record IDs, atomic numbers, charges, and multiplicities matched. The
largest coordinate difference caused by JSON floating-point representation was
`2.22e-16 Angstrom`. Because absolute PFP and GPU4PySCF energies have different
element/composition reference offsets, only same-composition-centered energy
errors are interpreted.

Aggregate PFP errors over all 50 records:

| Metric | Value |
|---|---:|
| Same-composition-centered energy MAE | 0.03837 eV |
| Same-composition-centered energy RMSE | 0.06550 eV |
| Force-component MAE | 0.07715 eV/Angstrom |
| Force-component RMSE | 0.13472 eV/Angstrom |
| Maximum absolute force-component error | 1.17191 eV/Angstrom |

The exact existing 40-train / 10-holdout parent split gives:

| Model | Partition | Centered energy MAE (eV) | Force MAE (eV/Angstrom) | Force max (eV/Angstrom) |
|---|---|---:|---:|---:|
| base UMA-S-1.2 | train | 0.00735 | 0.02546 | 0.25485 |
| PFP v9 R2SCAN+D3 | train | 0.04285 | 0.07811 | 1.17191 |
| 200-step fine-tuned UMA | train | 0.64606 | 0.93811 | 15.05486 |
| base UMA-S-1.2 | holdout (SiCl4) | 0.00059 | 0.00404 | 0.05421 |
| PFP v9 R2SCAN+D3 | holdout (SiCl4) | 0.02046 | 0.07272 | 0.39708 |
| 200-step fine-tuned UMA | holdout (SiCl4) | 0.36340 | 0.80598 | 7.70731 |

The 200-step result is the already-recorded failed overfit checkpoint and is
included only as a common-scale reference. It is not evidence about the newer
2,000/10,000-step experiments. On this narrow neutral-singlet displacement
set, base UMA is substantially closer to the GPU4PySCF teacher than PFP.

2026-09-01 addendum: the final 10,000-step checkpoint was subsequently
evaluated on the same split. It reached train centered-energy/force MAE of
0.000663 eV / 0.002528 eV/Angstrom, but holdout values were 0.044772 eV /
0.239682 eV/Angstrom. It therefore demonstrates train overfit and poor
composition-held-out transfer; it does not replace base UMA as the screening
baseline. Full provenance is recorded in
`2026-09-01_matlantis_multifidelity_mf0.md`.

The PFP discrepancy is strongly concentrated in larger perturbations:

| Displacement group | Records | Centered energy MAE (eV) | Force MAE (eV/Angstrom) | Force max (eV/Angstrom) |
|---|---:|---:|---:|---:|
| sigma=0.04 Angstrom | 40 | 0.00629 | 0.05490 | 0.22891 |
| sigma=0.12 Angstrom | 10 | 0.06386 | 0.16616 | 1.17191 |

The largest discrepancy is
`engineering_50_v1_h3si_gecl3_seed_disp0p12_s2026083508`, with force-component
maximum absolute error 1.17191 eV/Angstrom. The strong displacement dependence
means the current 50-point result should not be summarized by one aggregate
number alone.

## Fine-tuning decision

PFP energies and forces should **not** be appended directly to the canonical
GPU4PySCF teacher dataset:

1. the theory levels differ;
2. PFP labels are not charge/spin-state resolved;
3. the measured discrepancies are geometry dependent, especially for the
   `sigma=0.12 Angstrom` structures, so an element reference shift cannot
   reconcile the force targets or all relative energies; and
4. the current canonical record and ASE-LMDB schemas intentionally identify
   GPU4PySCF teacher provenance.

The useful near-term route is to use PFP for **candidate generation rather than
teacher labeling**. PFP optimization, MD, or reaction-path sampling can cheaply
propose neutral geometries; UMA/PFP disagreement and geometry filters can rank
them; selected structures are then relabeled with the pinned GPU4PySCF protocol
before entering UMA fine-tuning. This adds structural coverage without mixing
potential-energy surfaces.

A second, research-only option is a separately identified multi-fidelity or
distillation dataset with an auxiliary loss/head. It requires trainer and
schema support plus an ablation against the GPU4PySCF-only baseline; the current
pipeline must continue to reject PFP records as canonical labels.

## Provenance

| Artifact | SHA-256 |
|---|---|
| split manifest | `9a5c4c384835618756380c606ba48d6401ed3ae35f8444b912238fdc75cf0496` |
| base UMA evaluation | `52eaf82d7d34a9e954a06a8061b27936e45c294db90810a7c67ab3450f19c9e3` |
| 200-step UMA evaluation | `b741f2f81872185e34d4825e908b618ac27a9f3f73aa84be64bee06fbb2f654c` |
| PFP run identity | `9c0d75175717b29bae1b6cbf855b1530f2f1e88b99f41e7a5b01032e4e1f36e9` |
| PFP/GPU4PySCF comparison summary | `62a21024f5f23dcfc73d6a8e1e2b30c648a20e75eb2a3fcfec3211f36af13df4` |
| PFP/GPU4PySCF per-record CSV | `e544ce11ffbfb210e47d4492a88fd2c499a7930c3ced46d74576d68fb79f818b` |

The machine-readable tracked model comparison is
`validation/results_analysis/matlantis_pfp_engineering_50_model_comparison.csv`.
