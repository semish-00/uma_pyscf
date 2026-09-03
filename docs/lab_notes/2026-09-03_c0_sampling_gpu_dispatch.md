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

## Next actions

1. Freeze C0 exit evidence: aggregate label/failure accounting, CPU/GPU
   sentinels, QC v2 distance/severity policy, and cheap-signal calibration.
2. After the C0 exit checks pass, freeze a non-overlapping T0 before model-aware
   acquisition or retraining. Do not start bulk P0/P1 before the plan gates.
