# C0 sampling and multi-GPU label dispatch — 2026-09-03

## Outcome

- C0 baseline before this run: 117/180 accepted GPU4PySCF labels.
- Base-UMA moderate-temperature candidate generation is running as Slurm job
  `1825031`; its target is a blind 27-record portfolio.
- A controlled strong-distortion pool generated 18/18 geometry-QC-passing
  candidates across six neutral-singlet parents.
- The strong-distortion preparation job `1825041` completed and froze the
  candidate and portfolio manifests with checksums.
- GPU4PySCF label array `1825042_[0-7]` completed all eight deterministic,
  cost-balanced execution shards. All 18 DFT calculations converged; provisional
  QC accepted 17 and retained one boundary rejection. This raises the
  conservative C0 accepted count from 117 to 134/180.

## Scientific handling

The distortion factors are 0.70, 0.78, and 1.22 on one terminal M-H or M-Cl
bond per parent. Candidate generation rejects overlap/fragment errors but does
not reject large forces or high energies. Teacher convergence and electronic
QC decide whether a geometry becomes `valid_high_energy`, is quarantined as
`electronic_ambiguous`, or is rejected as `hard_invalid`.

The retained boundary case is
`c0_strong_distortion_pool_v1_si2h5cl_bond02_x0p7`. Its maximum gradient is
1.0066986 hartree/bohr against the provisional 1.0 threshold, while the SCF,
protocol, geometry, and gradient-norm checks pass. It is not silently promoted
or discarded: the raw converged label remains available for the high-energy
boundary audit, and the current accepted count excludes it.

On 2026-09-04, the moderate-MD primary run retained 30 finite trajectories but
could select only 21 records under the original five-per-parent cap because the
Ge2H4Cl2 parent had no accepted temperature-ratio trajectory. A parent-specific
recovery run (`1825383`) used lower thermostat targets and produced 8/8 finite
trajectories and 40/40 candidate frames. The combined blind portfolio selected
21 primary and 6 recovery records, kept all six parents, and capped each
trajectory at one record. Label array `1825468_[0-7]` converged and provisional
QC accepted all 27 records.

The original Si2H5Cl 0.70 compression remains in the boundary ledger. A fixed
0.72 compression replacement was generated without changing the global QC
threshold; jobs `1825484` and `1825485` produced one converged, QC-accepted
label. The high-energy quota is therefore 18/18 accepted while retaining the
original boundary raw label, and conservative C0 progress is 162/180.

The final curated graph-edit tranche used six manually reviewed,
valence-saturated neutral-singlet dimer topologies and positional isomers. Three
fixed 0.03 angstrom Cartesian displacements per parent gave 18/18 geometry-QC
candidates. Preparation job `1825503` froze the blind manifest and checksums;
label array `1825504_[0-7]` completed all shards with 18 raw records, no failure
files, and 18/18 provisional QC acceptance. C0 therefore reaches 180/180 while
preserving all source quotas and without using acquisition scores.

The independent-reaction comparison shows that the 10,000-step engineering
model is overfit and is not a valid acquisition model. Holdout-guided follow-up
will use base-UMA residuals only on diagnostic C0/P0 groups, preserve complete
reaction/parent grouping, and exclude the known ambiguous SiHCl3 detached-H
point until a separate state/stability audit resolves it.

## Base UMA retrospective signal

Jobs `1825518`–`1825523` ran base UMA-S-1.2 independently on the six frozen C0
source manifests. All jobs completed and produced 180 predictions with input
checksums. Teacher gradients were converted to forces and relative-energy errors
were centered within each parent/reaction group. The known detached-H record
`c0_independent_reaction_paths_36_v1_sihcl3_to_sicl2_hcl_f00003` was reported
separately and excluded from trusted acquisition diagnostics.

| source | trusted records | relative-energy MAE (eV) | force RMSE (eV/A) |
|---|---:|---:|---:|
| local | 45 | 0.0051 | 0.0471 |
| internal scan | 36 | 0.0478 | 0.1045 |
| independent path | 35 | 0.0206 | 0.0825 |
| moderate MD | 27 | 0.0049 | 0.0325 |
| strong distortion | 18 | 0.0102 | 0.0878 |
| curated graph edit | 18 | 0.0020 | 0.0316 |

The detached-H record alone has force RMSE 0.2248 eV/A. It remains
`electronic_ambiguous` and is not a resampling seed. Internal scans are the
largest trusted source-level gap in both relative energy and forces;
strong-distortion and reaction-path forces are the next diagnostic targets.

## Equal-budget diagnostic shell comparison

Three 12-record arms used the same 0.02/0.05 angstrom shell budget, retained
parent/reaction grouping, and stayed outside the blind C0 180 and T0. All 36
teacher calculations converged. The random arm accepted 12/12; the residual arm
accepted 10/12; the residual-plus-source-diversity arm accepted 11/12. The three
QC rejections are Ge2H5Cl high-gradient boundary labels, not calculation or
geometry failures; raw labels remain preserved and the global threshold was not
changed.

| arm | accepted/raw | mean record force RMSE (eV/A) | records >=0.1 eV/A | pair-centered relative-energy MAE (eV) |
|---|---:|---:|---:|---:|
| source-stratified random | 12/12 | 0.0262 | 0/12 | 0.0039 |
| base-UMA residual | 10/12 | 0.2097 | 9/12 | 0.0252 |
| residual + source diversity | 11/12 | 0.1105 | 5/12 | 0.0096 |

The residual arm enriches the force-error tail by nine of twelve records versus
zero of twelve for the random arm at the diagnostic 0.1 eV/A cutoff. This is
retrospective C0 evidence, not a frozen production acquisition guarantee. The
source-diverse arm excluded the entire SiHCl3 reaction group conservatively
rather than sampling next to the known detached-H artifact.

## GPU direct-SCF sentinel

Six source-stratified neutral-singlet records were recomputed without density
fitting in Slurm jobs `1826000` and `1826001`. All six direct calculations
converged on the first attempt. Against the frozen production density-fit
records at the same coordinates, the largest absolute energy difference was
1.7521 meV, the largest gradient-component RMSE was 0.0009374 eV/angstrom, and
the largest absolute gradient-component difference was 0.0017927
eV/angstrom. Raw records, attempt ledgers, protocol/input checksums, and the
non-release status are retained under
`/lustre/user140002/runs/calibration/c0_direct_sentinel_v1/20260904_v1`.
These descriptive results do not relax or newly freeze any global QC
threshold.

The direct comparison was then extended to the frozen 12-record base-UMA
residual shell, concentrating the audit on internal-scan and strong-distortion
boundaries without changing a C0 source quota. Slurm arrays `1826591` and
`1826592` completed all 12 records on the first direct attempt with no failure.
The largest absolute density-fit difference was 3.2235 meV in energy, 0.001124
eV/angstrom in gradient-component RMSE, and 0.003194 eV/angstrom in the largest
gradient component. The raw direct and density-fit record checksums and the
descriptive comparison are retained at
`/lustre/user140002/runs/calibration/c0_direct_boundary_sentinel_v1`; the
comparison report SHA-256 is
`c4a924190b3d6e41c51cccef9baf57957711d9aefbdf52e6c3b943677a96a490`.
The two high-gradient Ge2H5Cl density-fit labels remain preserved as boundary
records; direct parity does not convert their existing QC disposition into
automatic acceptance.

## Next actions

1. Freeze C0 exit evidence: aggregate label/failure accounting, CPU/GPU
   sentinels, QC v2 distance/severity policy, and cheap-signal calibration.
2. After the C0 exit checks pass, freeze a non-overlapping T0 before model-aware
   acquisition or retraining. Do not start bulk P0/P1 before the plan gates.
